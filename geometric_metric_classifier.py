# geometric_gmc_classifier_v3.py
from __future__ import annotations
from typing import Optional, Dict, Union, Any
import copy
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

Array = np.ndarray

# ---------- numerics ----------
def _logsumexp(a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    a = np.asarray(a)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def _softmax(S: Array, axis: int = 1) -> Array:
    M = np.max(S, axis=axis, keepdims=True)
    e = np.exp(S - M)
    return e / np.sum(e, axis=axis, keepdims=True)

def _global_l2_norm(dict_arrs: Dict[Any, Array]) -> float:
    tot = 0.0
    for _, v in dict_arrs.items():
        tot += float(np.sum(v * v))
    return float(np.sqrt(tot + 1e-30))

# ---------- model ----------
class GeometricMixtureClassifier(BaseEstimator, ClassifierMixin):
    """
    GMC: per-class mixture of hyperplanes with soft-OR (LSE) pooling and across-class softmax.
    Add-ons:
      - feature maps (RFF) for non-linear lift
      - silhouette-based auto selection of M_c
      - robust inits: kmeans / logreg / random
    """
    def __init__(
        self,
        # capacity
        n_planes: Union[int, Dict[Any, int], str] = 3,  # int | dict | "auto"
        alpha: float = 6.0,
        # optimization
        lr: float = 5e-2,
        max_epochs: int = 120,
        batch_size: int = 256,
        # regularization
        l2: Union[float, Dict[Any, float]] = 1e-4,
        unused_penalty: float = 0.5,  # usage-aware L2: lambda*(1+beta/usage)
        # preprocessing
        standardize: bool = True,
        auto_pca: bool = True,
        pca_variance: float = 0.95,
        pca_whiten: bool = False,
        # RFF control
        use_rff: str = "auto",               # {"off","on","auto"}  <-- public knob
        rff_dim: int = 512,                  # base dim D -> output is 2D if cos+sin
        rff_gamma: Union[str, float] = "auto",  # "auto" or positive float
        # legacy (kept for compatibility; ignored if use_rff != "auto")
        feature_map: str = "none",           # {"none","rff"}
        # init & schedules
        jitter_sigma: Union[float, str] = "auto",
        init_mode: str = "auto",             # {"auto","kmeans","logreg","random"}
        alpha_anneal: bool = True,
        alpha_start: Optional[float] = None,
        lr_schedule: str = "cosine",
        exp_gamma: float = 0.98,
        # loss shaping
        label_smoothing: float = 0.02,
        class_weight: Union[str, Dict[Any, float], None] = "auto",
        # early stop
        early_stopping: bool = True,
        val_split: float = 0.1,
        patience: int = 12,
        # stability
        grad_clip_norm: Optional[float] = 5.0,
        max_weight_norm: Optional[float] = None,
        # plane selection
        plane_select: str = "fixed",         # {"fixed","silhouette"}
        silhouette_kmax: int = 6,
        silhouette_metric: str = "euclidean",
        # misc
        random_state: Optional[int] = 0,
        verbose: int = 1,
    ):
        # core
        self.n_planes = n_planes
        self.alpha = float(alpha)
        # opt
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        # reg
        self.l2 = l2
        self.unused_penalty = float(unused_penalty)
        # preprocessing
        self.standardize = bool(standardize)
        self.auto_pca = bool(auto_pca)
        self.pca_variance = float(pca_variance)
        self.pca_whiten = bool(pca_whiten)
        # RFF controls
        self.use_rff = str(use_rff).lower()          # public mode
        self.rff_dim = int(rff_dim)
        self.rff_gamma = rff_gamma
        # legacy (ignored if use_rff != "auto")
        self.feature_map = feature_map
        # inits & schedules
        self.jitter_sigma = jitter_sigma
        self.init_mode = init_mode
        self.alpha_anneal = bool(alpha_anneal)
        self.alpha_start = alpha_start
        self.lr_schedule = lr_schedule
        self.exp_gamma = float(exp_gamma)
        # loss shaping
        self.label_smoothing = float(label_smoothing)
        self.class_weight = class_weight
        # early stop
        self.early_stopping = bool(early_stopping)
        self.val_split = float(val_split)
        self.patience = int(patience)
        # stability
        self.grad_clip_norm = grad_clip_norm
        self.max_weight_norm = max_weight_norm
        # plane selection
        self.plane_select = plane_select
        self.silhouette_kmax = int(silhouette_kmax)
        self.silhouette_metric = silhouette_metric
        # misc
        self.random_state = random_state
        self.verbose = int(verbose)

        # state
        self.classes_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        # resolved RFF
        self.feature_map_: str = "none"   # {"none","rff"} after fit
        self.rff_dim_: int = 0
        self.rff_gamma_: Optional[float] = None
        self.rff_Omega_: Optional[Array] = None
        self.rff_b_: Optional[Array] = None

        self.W_: Dict[Any, Array] = {}
        self.b_: Dict[Any, Array] = {}
        self.mW_: Dict[Any, Array] = {}
        self.vW_: Dict[Any, Array] = {}
        self.mb_: Dict[Any, Array] = {}
        self.vb_: Dict[Any, Array] = {}

        self._t = 0
        self._epoch = 0
        self._alpha_curr = self.alpha
        self.n_planes_resolved_: Dict[Any, int] = {}

    # ---------- helpers ----------
    def _alpha_for_epoch(self, epoch: int) -> float:
        if not self.alpha_anneal:
            return self.alpha
        a0 = self.alpha_start if (self.alpha_start is not None) else min(3.0, self.alpha)
        T = max(1, self.max_epochs - 1)
        return float(a0 + (self.alpha - a0) * (epoch / T))

    def _auto_class_weight(self, y: Array) -> Dict[Any, float]:
        cls, counts = np.unique(y, return_counts=True)
        inv = 1.0 / (counts.astype(float) + 1e-12)
        inv *= (len(counts) / inv.sum())
        return {c: float(w) for c, w in zip(cls, inv)}

    def _l2_for(self, c):
        if isinstance(self.l2, dict):
            vals = list(self.l2.values())
            default = float(np.mean(vals)) if len(vals) else 1e-4
            return float(self.l2.get(c, default))
        return float(self.l2)

    def _resolve_planes_by_silhouette(self, Z: Array, y: Array) -> Dict[Any, int]:
        rng = check_random_state(self.random_state)
        out: Dict[Any, int] = {}
        for c in np.unique(y):
            Zc = Z[y == c]
            best_k, best_s = 1, -1.0
            kmax = max(1, self.silhouette_kmax)
            for k in range(1, kmax + 1):
                if len(Zc) < max(2, k):
                    continue
                try:
                    km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                    labels = km.fit_predict(Zc)
                    s = -1.0 if k == 1 else silhouette_score(Zc, labels, metric=self.silhouette_metric)
                except Exception:
                    s = -1.0
                if s > best_s:
                    best_s, best_k = s, k
            out[c] = int(max(1, best_k))
        return {c: max(1, k) for c, k in out.items()}

    def _planes_for(self, c):
        # resolved dict first
        if self.n_planes_resolved_:
            return int(self.n_planes_resolved_.get(c, max(2, int(np.mean(list(self.n_planes_resolved_.values()))))))
        # user-provided dict
        if isinstance(self.n_planes, dict):
            if len(self.n_planes) == 0:
                return 2
            default = max(2, int(np.mean(list(self.n_planes.values()))))
            return int(self.n_planes.get(c, default))
        # fixed integer
        if isinstance(self.n_planes, int):
            return int(self.n_planes)
        # "auto" but not resolved yet (should be resolved in fit)
        return 2

    # ---------- RFF helpers ----------
    def _estimate_gamma(self, Z: Array) -> float:
        """Robust median-distance heuristic for RBF gamma. Uses scipy.pdist if available."""
        n = Z.shape[0]
        rng = check_random_state(self.random_state)
        m = min(n, 1024)
        idx = rng.choice(n, size=m, replace=False)
        A = Z[idx]

        try:
            from scipy.spatial.distance import pdist
            d2 = pdist(A, metric='sqeuclidean')
            med = float(np.median(d2)) if d2.size else 1.0
        except Exception:
            # NumPy fallback on a small subset to avoid O(m^2) explosion
            k = min(m, 256)
            A = A[:k]
            # compute upper-triangular pairwise sq distances in chunks
            med_candidates = []
            for i in range(k):
                # vectorized vs all j>i
                if i+1 < k:
                    diff = A[i+1:] - A[i]
                    d2_i = np.sum(diff*diff, axis=1)
                    med_candidates.append(np.median(d2_i))
            med = float(np.median(med_candidates)) if med_candidates else 1.0

        gamma = 1.0 / max(1e-6, med)
        return gamma

    def _maybe_fit_rff(self, Z: Array):
        """Fit RFF params if requested via use_rff."""
        # resolve final mode
        mode = self.use_rff
        if mode not in {"off", "on", "auto"}:
            mode = "off"

        # legacy "feature_map" only matters if use_rff is "auto"
        if mode == "auto":
            mode = "on" if str(self.feature_map).lower() == "rff" else "off"

        self.feature_map_ = "none"
        self.rff_dim_ = 0
        self.rff_gamma_ = None
        self.rff_Omega_ = None
        self.rff_b_ = None

        if mode != "on":
            return  # nothing to fit

        D = int(self.rff_dim)
        if D <= 0:
            return

        # gamma
        if isinstance(self.rff_gamma, str) and self.rff_gamma.lower() == "auto":
            gamma = self._estimate_gamma(Z)
        else:
            gamma = float(self.rff_gamma)

        rng = check_random_state(self.random_state)
        d = Z.shape[1]
        Omega = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, D))
        b = rng.uniform(0.0, 2.0 * np.pi, size=(D,))

        self.feature_map_ = "rff"
        self.rff_dim_ = int(D)
        self.rff_gamma_ = float(gamma)
        self.rff_Omega_ = Omega
        self.rff_b_ = b

    def _apply_rff(self, Z: Array) -> Array:
        if self.feature_map_ != "rff":
            return Z
        U = Z @ self.rff_Omega_ + self.rff_b_
        c = np.cos(U); s = np.sin(U)
        return np.sqrt(2.0 / self.rff_dim_) * np.concatenate([c, s], axis=1)

    # ---------- preprocessing ----------
    def _prep(self, X: Array, fit: bool, training_mode: bool = False) -> Array:
        """Full pipeline (standardize, PCA, then optional RFF depending on resolved self.use_rff)."""
        Z = np.asarray(X)

        if fit:
            if self.standardize:
                self.scaler_ = StandardScaler()
                Z = self.scaler_.fit_transform(Z)
            else:
                self.scaler_ = None

            use_pca = self.auto_pca and (Z.shape[1] > 50)
            if use_pca or (not self.auto_pca and self.pca_variance is not None):
                self.pca_ = PCA(
                    n_components=self.pca_variance,
                    svd_solver="full",
                    whiten=self.pca_whiten,
                    random_state=self.random_state,
                )
                Z = self.pca_.fit_transform(Z)
            else:
                self.pca_ = None

            if training_mode:
                if isinstance(self.jitter_sigma, str) and self.jitter_sigma == "auto":
                    sigma = 0.01 if Z.shape[1] > 50 else 0.0
                else:
                    sigma = float(self.jitter_sigma)
                if sigma > 0:
                    rng = check_random_state(self.random_state)
                    Z = Z + rng.normal(0, sigma, size=Z.shape)

            # Fit RFF params if requested (use_rff resolved by now)
            self._maybe_fit_rff(Z)
            Z = self._apply_rff(Z)

        else:
            if self.scaler_ is not None:
                Z = self.scaler_.transform(Z)
            if self.pca_ is not None:
                Z = self.pca_.transform(Z)
            Z = self._apply_rff(Z)

        return Z

    # ---------- inits ----------
    def _init_params_random(self, d: int) -> None:
        rng = check_random_state(self.random_state)
        self.mW_.clear(); self.vW_.clear(); self.mb_.clear(); self.vb_.clear()
        self.W_.clear(); self.b_.clear()

        for c in self.classes_:
            M = self._planes_for(c)
            self.W_[c] = 0.01 * rng.randn(M, d)
            self.b_[c] = np.zeros(M)
            self.mW_[c] = np.zeros_like(self.W_[c]); self.vW_[c] = np.zeros_like(self.W_[c])
            self.mb_[c] = np.zeros_like(self.b_[c]); self.vb_[c] = np.zeros_like(self.b_[c])
        self._t = 0

    def _init_params_kmeans(self, Z: Array, y: Array) -> None:
        rng = check_random_state(self.random_state)
        self.mW_.clear(); self.vW_.clear(); self.mb_.clear(); self.vb_.clear()
        self.W_.clear(); self.b_.clear()

        d = Z.shape[1]
        global_mean = np.mean(Z, axis=0)

        for c in self.classes_:
            M = self._planes_for(c)
            Zc = Z[y == c]
            if len(Zc) < max(2, M):
                Wc = 0.01 * rng.randn(M, d)
                bc = np.zeros(M)
            else:
                try:
                    km = KMeans(n_clusters=M, n_init=10, random_state=self.random_state)
                    labels = km.fit_predict(Zc)
                    centers = km.cluster_centers_
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"[init:kmeans] failed ({e}); falling back to random.")
                    Wc = 0.01 * rng.randn(M, d)
                    bc = np.zeros(M)
                    self.W_[c] = Wc; self.b_[c] = bc
                    self.mW_[c] = np.zeros_like(Wc); self.vW_[c] = np.zeros_like(Wc)
                    self.mb_[c] = np.zeros_like(bc); self.vb_[c] = np.zeros_like(bc)
                    continue

                Wc = np.zeros((M, d)); bc = np.zeros(M)
                for m in range(M):
                    mu = centers[m]
                    dir_vec = mu - global_mean
                    norm = np.linalg.norm(dir_vec) + 1e-12
                    w = dir_vec / norm
                    b = -np.dot(w, mu) + 0.5
                    Wc[m] = 0.01 * rng.randn(d) + w
                    bc[m] = b

            self.W_[c] = Wc; self.b_[c] = bc
            self.mW_[c] = np.zeros_like(Wc); self.vW_[c] = np.zeros_like(Wc)
            self.mb_[c] = np.zeros_like(bc); self.vb_[c] = np.zeros_like(bc)
        self._t = 0

    def _init_params_logreg(self, Z: Array, y: Array) -> None:
        """One-vs-rest logistic direction per class, replicated across planes with noise."""
        rng = check_random_state(self.random_state)
        self.mW_.clear(); self.vW_.clear(); self.mb_.clear(); self.vb_.clear()
        self.W_.clear(); self.b_.clear()

        d = Z.shape[1]
        for c in self.classes_:
            y_bin = (y == c).astype(int)
            lr = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs")
            try:
                lr.fit(Z, y_bin)
                w_dir = lr.coef_.reshape(-1)
                b_dir = float(lr.intercept_.reshape(()))
                wn = np.linalg.norm(w_dir) + 1e-12
                w_dir = w_dir / wn
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[init:logreg] failed ({e}); falling back to random dir.")
                w_dir = rng.randn(d); w_dir /= (np.linalg.norm(w_dir) + 1e-12)
                b_dir = 0.0

            M = self._planes_for(c)
            Wc = np.zeros((M, d)); bc = np.zeros(M)
            for m in range(M):
                Wc[m] = w_dir + 0.01 * rng.randn(d)
                bc[m] = b_dir + 0.01 * rng.randn()
            self.W_[c] = Wc; self.b_[c] = bc
            self.mW_[c] = np.zeros_like(Wc); self.vW_[c] = np.zeros_like(Wc)
            self.mb_[c] = np.zeros_like(bc); self.vb_[c] = np.zeros_like(bc)
        self._t = 0

    def _init_params(self, Z: Array, y: Array) -> None:
        d = Z.shape[1]
        mode = str(self.init_mode).lower()
        if mode == "auto":
            # try kmeans -> logreg -> random
            try:
                self._init_params_kmeans(Z, y)
                return
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[init:auto] KMeans failed ({e}); trying logreg.")
            try:
                self._init_params_logreg(Z, y)
                return
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[init:auto] logreg failed ({e}); falling back to random.")
            self._init_params_random(d); return

        if mode == "kmeans":
            try:
                self._init_params_kmeans(Z, y); return
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[init:kmeans] failed ({e}); falling back to random.")
                self._init_params_random(d); return

        if mode == "logreg":
            try:
                self._init_params_logreg(Z, y); return
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[init:logreg] failed ({e}); falling back to random.")
                self._init_params_random(d); return

        self._init_params_random(d)

    # ---------- forward ----------
    def _plane_logits(self, Z: Array) -> Dict[Any, Array]:
        out = {}
        for c in self.classes_:
            out[c] = Z @ self.W_[c].T + self.b_[c][None, :]
        return out

    def _class_scores_from_plane_logits(self, Z_by_class: Dict[Any, Array], alpha: float) -> Array:
        S_list = []
        for c in self.classes_:
            Zc = Z_by_class[c]
            s_c = (1.0 / alpha) * _logsumexp(alpha * Zc, axis=1)
            S_list.append(s_c)
        return np.stack(S_list, axis=1)

    def _forward_scores(self, Z: Array, alpha: float) -> Array:
        return self._class_scores_from_plane_logits(self._plane_logits(Z), alpha)

    # ---------- loss & grads ----------
    def _loss_and_grads(
        self,
        Z: Array,
        y_idx: Array,
        class_to_idx: Dict[Any, int],
        alpha: float,
        class_w: Optional[Dict[Any, float]],
    ):
        n = Z.shape[0]
        C = len(self.classes_)
        eps = self.label_smoothing

        Z_by_class = self._plane_logits(Z)
        S = self._class_scores_from_plane_logits(Z_by_class, alpha)
        P = _softmax(S, axis=1)

        true_probs = P[np.arange(n), y_idx]
        ce = -np.log(true_probs + 1e-12)

        if class_w is not None:
            w = np.array([class_w[self.classes_[yi]] for yi in y_idx], dtype=float)
            L_ce = float(np.mean(ce * w))
        else:
            w = np.ones_like(ce)
            L_ce = float(np.mean(ce))

        # dL/dS
        dS = P.copy()
        dS[np.arange(n), y_idx] -= 1.0
        dS /= float(n)
        dS *= w[:, None]

        gW = {}
        gb = {}
        reg = 0.0

        for j, c in enumerate(self.classes_):
            Zc = Z_by_class[c]
            Ac = _softmax(alpha * Zc, axis=1)  # responsibilities within class
            dZc = dS[:, [j]] * Ac              # credit assignment to planes

            gW_c = dZc.T @ Z
            gb_c = dZc.sum(axis=0)

            base_l2 = self._l2_for(c)
            if self.unused_penalty > 0.0:
                usage = Ac.mean(axis=0) + 1e-8
                l2_per = base_l2 * (1.0 + self.unused_penalty / usage)
                gW_c += (l2_per[:, None] * self.W_[c])
                reg += float(0.5 * np.sum(l2_per[:, None] * (self.W_[c] * self.W_[c])))
            else:
                gW_c += base_l2 * self.W_[c]
                reg += float(0.5 * base_l2 * np.sum(self.W_[c] * self.W_[c]))

            gW[c] = gW_c
            gb[c] = gb_c

        L = L_ce + reg
        return L, gW, gb, P

    # ---------- optimizer ----------
    def _adam_step(self, gW: Dict[Any, Array], gb: Dict[Any, Array],
                   beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        # gradient clipping (global)
        if self.grad_clip_norm is not None:
            gwn = _global_l2_norm(gW)   # returns L2 norm
            gbn = _global_l2_norm(gb)   # returns L2 norm
            gn = np.sqrt(gwn**2 + gbn**2)
            if gn > self.grad_clip_norm:
                scale = self.grad_clip_norm / (gn + 1e-12)
                for c in gW:
                    gW[c] *= scale
                for c in gb:
                    gb[c] *= scale

        # LR schedule
        self._t += 1
        if self.lr_schedule == "exp":
            lr_sched = self.exp_gamma ** self._epoch
        elif self.lr_schedule == "cosine":
            import math
            progress = min(1.0, max(0.0, self._epoch / max(1, self.max_epochs - 1)))
            lr_sched = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr_sched = 1.0

        base = self.lr * lr_sched
        lr_t = base

        for c in self.classes_:
            # weights
            self.mW_[c] = beta1 * self.mW_[c] + (1 - beta1) * gW[c]
            self.vW_[c] = beta2 * self.vW_[c] + (1 - beta2) * (gW[c] * gW[c])
            mW_hat = self.mW_[c] / (1 - beta1 ** self._t)
            vW_hat = self.vW_[c] / (1 - beta2 ** self._t)
            self.W_[c] -= lr_t * mW_hat / (np.sqrt(vW_hat) + eps)

            if self.max_weight_norm is not None:
                norms = np.linalg.norm(self.W_[c], axis=1, keepdims=True) + 1e-12
                scale = np.minimum(1.0, self.max_weight_norm / norms)
                self.W_[c] *= scale

            # biases
            self.mb_[c] = beta1 * self.mb_[c] + (1 - beta1) * gb[c]
            self.vb_[c] = beta2 * self.vb_[c] + (1 - beta2) * (gb[c] * gb[c])
            mb_hat = self.mb_[c] / (1 - beta1 ** self._t)
            vb_hat = self.vb_[c] / (1 - beta2 ** self._t)
            self.b_[c] -= lr_t * mb_hat / (np.sqrt(vb_hat) + eps)

    # ---------- RFF auto-selection ----------
    def _choose_rff_mode(self, X: Array, y: Array, X_val: Array, y_val: Array) -> str:
        """Return 'off' or 'on' by training two cheap probes and comparing val accuracy."""
        if self.verbose >= 1:
            print("[auto-RFF] probing 'off' vs 'on'...")

        def _mk_probe(use_rff_flag: str):
            pr = copy.deepcopy(self)
            pr.use_rff = use_rff_flag          # <-- CRITICAL: break recursion
            pr.feature_map = "rff" if use_rff_flag == "on" else "none"
            pr.max_epochs = min(30, max(10, self.max_epochs // 4))
            pr.patience = min(6, max(3, self.patience // 2))
            pr.verbose = max(0, self.verbose - 1)
            return pr

        # Linear probe
        probe_lin = _mk_probe("off")
        probe_lin.fit(X, y, X_val=X_val, y_val=y_val)
        acc_lin = accuracy_score(y_val, probe_lin.predict(X_val))

        # RFF probe
        probe_rff = _mk_probe("on")
        probe_rff.fit(X, y, X_val=X_val, y_val=y_val)
        acc_rff = accuracy_score(y_val, probe_rff.predict(X_val))

        if self.verbose >= 1:
            print(f"[auto-RFF] val acc: off={acc_lin:.4f}  on={acc_rff:.4f}")

        # prefer RFF only if it helps; small margin to avoid flapping
        if acc_rff >= acc_lin + 1e-3:
            return "on"
        return "off"

    # ---------- API ----------
    def fit(self, X: Array, y: Array, X_val: Optional[Array] = None, y_val: Optional[Array] = None):
        X = np.asarray(X); y = np.asarray(y)

        # --- validation handling (strict) ---
        if (X_val is None) ^ (y_val is None):
            raise ValueError("Provide both X_val and y_val, or neither.")
        use_external_val = (X_val is not None) and (y_val is not None)

        # Decide RFF mode if 'auto' using a quick probe on *raw* inputs (pipeline inside probes).
        if self.use_rff == "auto":
            # Build a small validation if none was provided
            if not use_external_val:
                X_train, X_hold, y_train, y_hold = train_test_split(
                    X, y, test_size=self.val_split, stratify=y, random_state=self.random_state
                )
            else:
                X_train, y_train = X, y
                X_hold, y_hold = np.asarray(X_val), np.asarray(y_val)

            decided = self._choose_rff_mode(X_train, y_train, X_hold, y_hold)
            if self.verbose >= 1:
                print(f"[auto-RFF] selected mode: {decided}")
            self.use_rff = decided  # lock in
        # else: use_rff is fixed by user

        # Preprocess (fit) with the now-locked RFF mode
        Z = self._prep(X, fit=True, training_mode=True)

        # classes + weights
        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[yy] for yy in y], dtype=int)

        if isinstance(self.class_weight, dict):
            class_w = self.class_weight
        elif self.class_weight == "auto":
            class_w = self._auto_class_weight(y)
        else:
            class_w = None

        # resolve planes if requested
        self.n_planes_resolved_.clear()
        need_auto_mc = (isinstance(self.n_planes, str) and self.n_planes.lower() == "auto") \
                       or (str(self.plane_select).lower() == "silhouette")
        if need_auto_mc:
            self.n_planes_resolved_ = self._resolve_planes_by_silhouette(Z, y)
            if self.verbose >= 1:
                print("[auto M_c]", self.n_planes_resolved_)

        # init
        self._init_params(Z, y)

        # internal validation split if needed
        if self.early_stopping:
            if use_external_val:
                Z_val = self._prep(np.asarray(X_val), fit=False)
                y_val_idx = np.array([class_to_idx[yy] for yy in y_val], dtype=int)
            else:
                Z, Z_val, y_idx, y_val_idx = train_test_split(
                    Z, y_idx, test_size=self.val_split, stratify=y_idx, random_state=self.random_state
                )

        # train loop
        best_state = None
        best_val = float("inf")
        wait = 0
        rng = check_random_state(self.random_state)
        n = Z.shape[0]

        for epoch in range(self.max_epochs):
            self._epoch = epoch
            self._alpha_curr = self._alpha_for_epoch(epoch)

            idx = rng.permutation(n)
            total_loss = 0.0
            batches = 0

            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Zb, yb = Z[batch], y_idx[batch]
                L, gW, gb, _ = self._loss_and_grads(Zb, yb, class_to_idx, self._alpha_curr, class_w)
                self._adam_step(gW, gb)
                total_loss += L; batches += 1

            train_loss = total_loss / max(1, batches)

            if self.early_stopping:
                Zb = Z_val; yb = y_val_idx
                S_val = self._forward_scores(Zb, self._alpha_curr)
                P_val = _softmax(S_val, axis=1)
                true_p = P_val[np.arange(len(yb)), yb]
                ce = -np.log(true_p + 1e-12)
                if class_w is not None:
                    wv = np.array([class_w[self.classes_[yy]] for yy in yb], dtype=float)
                    val_loss = float(np.mean(ce * wv))
                else:
                    val_loss = float(np.mean(ce))

                if self.verbose >= 1:
                    print(f"[epoch {epoch+1:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  alpha={self._alpha_curr:.3f}")
                if self.verbose >= 2:
                    usage_log = {}
                    Z_by_cls = self._plane_logits(Z)
                    for c, Zc in Z_by_cls.items():
                        usage_log[int(c)] = np.round(_softmax(self._alpha_curr * Zc, axis=1).mean(axis=0), 4).tolist()
                    print("[usage]", usage_log)

                if val_loss + 1e-9 < best_val:
                    best_val = val_loss
                    best_state = (
                        copy.deepcopy(self.W_), copy.deepcopy(self.b_),
                        copy.deepcopy(self.mW_), copy.deepcopy(self.vW_),
                        copy.deepcopy(self.mb_), copy.deepcopy(self.vb_),
                        self._t, self._epoch, self._alpha_curr
                    )
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        if self.verbose >= 1:
                            print(f"[early stopping] restoring best (val_loss={best_val:.6f})")
                        if best_state is not None:
                            (self.W_, self.b_, self.mW_, self.vW_,
                             self.mb_, self.vb_, self._t, self._epoch, self._alpha_curr) = best_state
                        break
            else:
                if self.verbose >= 1:
                    print(f"[epoch {epoch+1:03d}] train_loss={train_loss:.6f}  alpha={self._alpha_curr:.3f}")

        return self

    def decision_function(self, X: Array) -> Array:
        Z = self._prep(np.asarray(X), fit=False)
        return self._forward_scores(Z, self._alpha_curr)

    def predict_proba(self, X: Array) -> Array:
        S = self.decision_function(X)
        return _softmax(S, axis=1)

    def predict(self, X: Array) -> Array:
        S = self.decision_function(X)
        return self.classes_[np.argmax(S, axis=1)]

    def score(self, X: Array, y: Array) -> float:
        return accuracy_score(y, self.predict(X))

    # diagnostics
    def plane_scores(self, X: Array, class_label: Optional[Any] = None):
        Z = self._prep(np.asarray(X), fit=False)
        Z_by_class = self._plane_logits(Z)
        if class_label is None:
            return Z_by_class
        return Z_by_class[class_label]

    def plane_responsibilities(self, X: Array, class_label: Optional[Any] = None):
        Z_by_class = self.plane_scores(X, class_label=None)
        if class_label is None:
            out = {}
            for c, Zc in Z_by_class.items():
                out[c] = _softmax(self._alpha_curr * Zc, axis=1)
            return out
        else:
            Zc = Z_by_class[class_label]
            return _softmax(self._alpha_curr * Zc, axis=1)
