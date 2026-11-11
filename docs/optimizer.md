# Convex Quality Optimization & Covariance Estimation Architecture

## 1. Overview
The `src/optimizer.py` module supplies risk models and convex quadratic optimization routines for Milestone 3 of the Black CEO Portfolio backtesting platform. Because the active constituent universe $N_t$ is sparse (historically ranging between 1 and 8 firms), standard sample covariance matrix estimation suffers from extreme estimation error, eigenvalue dispersion, and near-singular conditioning. 

To ensure numerical stability and robust risk budgeting, the engine implements Ledoit-Wolf (2004) analytical shrinkage and spectral ridge regularizations.

---

## 2. Mathematical Formulation: Ledoit-Wolf Shrinkage

Given an active asset return matrix $X \in \mathbb{R}^{T \times N}$ across a historical lookback window $T$ (default: 252 sessions) and $N$ active assets:

### 2.1 Sample Covariance Matrix
Let $\bar{x} \in \mathbb{R}^N$ denote the sample mean return vector:
$$\bar{x} = \frac{1}{T} \sum_{t=1}^T x_t$$

The unbiased sample covariance matrix $S \in \mathbb{R}^{N \times N}$ is:
$$S = \frac{1}{T - 1} \sum_{t=1}^T (x_t - \bar{x})(x_t - \bar{x})^\top$$

When $N$ approaches $T$ (or when single-stock returns exhibit strong collinearity), the sample covariance matrix $S$ over-disperses eigenvalues: the largest eigenvalues are biased upward and the smallest eigenvalues are biased toward zero. Inverting $S$ in a Mean-Variance or Minimum-Variance optimization produces erratic and concentrated weights.

### 2.2 Shrinkage Target ($F$)
Ledoit & Wolf (2004) introduce a structured shrinkage target $F$ defined as the constant-correlation or scaled identity matrix. For our default formulation, we target average sample variance scaled across the identity:
$$\mu = \frac{1}{N} \text{Tr}(S)$$
$$F = \mu I_N$$

Where:
* $I_N$ is the $N \times N$ identity matrix.
* $\mu$ is the cross-sectional mean variance of the active universe.

### 2.3 Optimal Convex Shrinkage Intensity ($\delta^*$)
The regularized covariance matrix $\Sigma^*$ is a convex linear combination of the sample covariance matrix $S$ and the target $F$:
$$\Sigma^* = (1 - \delta^*) S + \delta^* F, \quad \delta^* \in [0, 1]$$

The optimal shrinkage intensity $\delta^*$ minimizes the expected quadratic loss under the Frobenius norm:
$$L(\delta) = \|\Sigma^* - \Sigma_{\text{true}}\|_F^2$$

Analytically, $\delta^*$ is estimated via:
$$\delta^* = \max\left(0, \min\left(1, \frac{\kappa}{T}\right)\right)$$

Where $\kappa$ measures the asymptotic variance of the sample covariance entries relative to the target distance.

---

## 3. Numerical Conditioning for Low-Sample Regimes (#151)

During transitions where an asset has recently entered the universe or historical data spans fewer than $T_{\min} = 63$ trading days, additional conditioning guarantees strict positive definiteness:

1. **Symmetric Projection:**
   $$\Sigma_{\text{sym}} = \frac{1}{2} (\Sigma^* + (\Sigma^*)^\top)$$
2. **Spectral Eigenvalue Clipping (Ridge Regularization):**
   Compute the spectral decomposition:
   $$\Sigma_{\text{sym}} = V \Lambda V^\top, \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$$
   Floor eigenvalues by a numerical buffer $\epsilon = 10^{-4}$:
   $$\tilde{\Lambda} = \text{diag}(\max(\lambda_i, \epsilon))$$
   Reconstruct:
   $$\tilde{\Sigma} = V \tilde{\Lambda} V^\top + \epsilon I_N$$
3. **Strict Positive Definiteness Invariant:**
   Guarantees for any non-zero allocation vector $w \neq 0$:
   $$w^\top \tilde{\Sigma} w > 0$$

---

## 4. Integration with CVXPY Solvers

In subsequent Day 13–16 routines, $\tilde{\Sigma}$ serves as the quadratic risk term in convex programs:
$$\min_{w} \quad \frac{1}{2} w^\top \tilde{\Sigma} w - q^\top w$$
$$\text{subject to} \quad \mathbf{1}^\top w = 1, \quad 0 \le w_i \le \min\left(0.35, \frac{2}{N_t}\right)$$

Because $\tilde{\Sigma} \succ 0$, the Hessian is strictly positive definite, guaranteeing unique global convergence under OSQP and CLARABEL interior-point solvers.

## Heuristic Benchmark: Inverse-Volatility Risk Parity Allocation

### 1. Motivation & Mathematical Formulation
Naive equal-weighting ($w_i = 1/N_t$) assumes identical variance across all portfolio assets. In an all-cap universe containing both large-cap blue chips and small/micro-cap equities, equal-weighting allows high-variance idiosyncratic noise to dominate portfolio return and volatility.

Inverse-volatility allocation budgets portfolio capital inversely to each active asset's trailing sample standard deviation. Over a rolling lookback window $L = 63$ trading sessions (approximately one business quarter):

$$\sigma_{i, t} = \sqrt{\frac{1}{L - 1} \sum_{\tau=t-L+1}^{t} \left( r_{i, \tau} - \bar{r}_{i, t} \right)^2}$$

### 2. Numerical Regularization ($\epsilon$ Floor)
To prevent division-by-zero errors or numerical instability when an asset exhibits zero or near-zero variance (e.g., newly listed assets or trading halts), an analytical floor $\epsilon = 10^{-4}$ is applied:

$$\tilde{\sigma}_{i, t} = \max(\sigma_{i, t}, \, \epsilon)$$

The unnormalized inverse-volatility weight for active constituent $i \in \mathcal{A}_t$ is:

$$w_{i, t}^{\text{raw}} = \frac{1}{\tilde{\sigma}_{i, t}}$$

### 3. Budget Normalization & Concentration Defense
Let $N_t = |\mathcal{A}_t|$ denote the number of active Black CEO constituents on date $t$. The total portfolio equity budget $B_t$ is determined by the concentration defense rule:

$$B_t = \begin{cases} 
0.0 & \text{if } N_t = 0 \\ 
\frac{1}{3} & \text{if } 1 \le N_t < 3 \\ 
1.0 & \text{if } N_t \ge 3 
\end{cases}$$

The final portfolio weight for asset $i$ is:

$$w_{i, t} = B_t \cdot \frac{w_{i, t}^{\text{raw}}}{\sum_{j \in \mathcal{A}_t} w_{j, t}^{\text{raw}}}$$

The remaining allocation $(1 - B_t)$ sits in risk-free cash, dampening drawdown risk during periods of universe sparsity.
