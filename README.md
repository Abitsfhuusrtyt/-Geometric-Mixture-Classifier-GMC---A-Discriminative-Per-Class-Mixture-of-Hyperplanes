# GMC: Geometric Multi-plane Classifier

## Problem Statement

Traditional classifiers (e.g., logistic regression, random forests, SVMs) either struggle with calibration or trade accuracy for efficiency. Kernel methods like RBF-SVM achieve strong accuracy but are slow at inference, while neural nets need large parameter counts to generalize well.

## Our Approach

We propose **Geometric Multi-plane Classifier (GMC)**, a lightweight, plane-based probabilistic model. GMC learns a set of linear separators (“planes”) that partition the feature space. Probabilities are estimated from the relative distances of samples to these planes, followed by **temperature scaling** for improved calibration.

### Derivation (sketch)

1. Define \$M\$ planes parameterized by \$\mathbf{w}\_i, b\_i\$.
2. Map input \$\mathbf{x}\$ to a representation via signed distances:
   $d_i(\mathbf{x}) = \mathbf{w}_i^\top \mathbf{x} + b_i$
3. Aggregate plane responses into a probability distribution using softmax:
   $P(y=k \mid \mathbf{x}) = \frac{\exp(-T \cdot f_k(\mathbf{x}))}{\sum_j \exp(-T \cdot f_j(\mathbf{x}))}$
   where \$T\$ is the learned temperature for calibration.

This yields **logistic-like probabilities** but with geometric interpretability and lower complexity than kernel expansions.

## Why GMC Beats Baselines

* **Accuracy:** Matches/bests RBF-SVM and MLP across datasets.
* **Calibration:** Lower Expected Calibration Error (ECE) after scaling than logistic regression, random forests, or GBMs.
* **Efficiency:** Inference latency is significantly lower than RBF-SVM at similar accuracy.
* **Simplicity:** Small parameter counts (few thousand) vs tens of thousands for neural nets.

## Recommended Defaults

* Variant: **linear** (RFF optional)
* `planes_total`: **6**
* `rff_dim`: **256**
* Temperature scaling: **enabled**


