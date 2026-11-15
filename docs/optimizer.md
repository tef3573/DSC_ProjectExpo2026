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


### 2.3 Optimal Convex Shrinkage Intensity ($\delta^{\ast}$)
The regularized covariance matrix $\Sigma^{\ast}$ is a convex linear combination of the sample covariance matrix $S$ and the structured target $F$:

$$\Sigma^{\ast} = (1 - \delta^{\ast}) S + \delta^{\ast} F, \quad \delta^{\ast} \in [0, 1]$$

The optimal shrinkage intensity $\delta^{\ast}$ minimizes the expected quadratic loss under the Frobenius norm:

$$L(\delta) = \|\Sigma^{\ast} - \Sigma_{\text{true}}\|_{F}^{2}$$

Analytically, $\delta^{\ast}$ is estimated via:

$$\delta^{\ast} = \max\left(0, \, \min\left(1, \, \frac{\kappa}{T}\right)\right)$$

where $\kappa = \frac{\pi - \rho}{\gamma}$ measures the asymptotic variance of the sample covariance entries relative to the squared distance to the shrinkage target $F$, and $T$ is the number of historical sample observations.

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

---

## 4. Convex Quadratic Program (QP) Formulation & Solver Architecture

### 4.1 Quadratic Program Specification (#162, #163, #164)
The minimum-variance portfolio allocation is formulated as a convex quadratic program over the decision variable vector $w \in \mathbb{R}^N$:

$$\min_{w} \quad \frac{1}{2} w^{\top} \Sigma^{*} w$$

$$\text{subject to} \quad w \ge 0, \quad \sum_{i=1}^{N} w_i = B_t$$

where $\Sigma^{*} \in \mathbb{S}_{+}^{N}$ is the Ledoit-Wolf shrunk, PSD-conditioned covariance matrix, and $B_t \in \{0, \frac{1}{3}, 1.0\}$ represents the active equity budget under the concentration defense rule.

### 4.2 Solver Selection Rationale: CLARABEL vs. OSQP (#165)

`ConvexPortfolioOptimizer` implements a dual-solver hierarchy designed for robust, zero-crash execution across historical market regimes:

---
```mermaid
flowchart TD
    A["    Input: Conditioned Covariance Matrix (PSD)     "] --> B[" Primary Solver: CLARABEL (IPM)<br/><i>Interior-Point Conic Optimization</i>"]
    B -->|Success| C["Optimal Weights"]
    B -->|Failure| D["Fallback Solver: OSQP<br/><i>First-Order ADMM Splitting</i>"]
    D -->|Success| C
    D -->|Failure| E["Deterministic Fallback<br/><i>Inverse-Vol / Equal Weight</i>"]

```
---

## 5. Fundamental Quality Score Formulation & Objective Tilts

### 5.1 Point-in-Time Quality Proxy Vector ($q_t$) (#168)
To avoid look-ahead and restatement bias across historical constituents (including delisted equities such as `JCP`, `WBA`, and `QRTEA`), the quality scoring vector $q_t \in \mathbb{R}^N$ is formulated strictly from point-in-time historical price action prior to trading session $t$:

1. **Risk-Adjusted Drift ($12\text{–}1$ Month Information Ratio):**
   Computed over the trailing 252-day window ending 21 trading days prior to $t$ ($[t-252, \, t-21)$) to capture persistent operational outperformance while avoiding short-term reversal contamination:
   $$S_{\text{drift}, i, t} = \frac{\mu_{i, [t-252, t-21)}}{\max(\sigma_{i, [t-252, t-21)}, \epsilon)}$$

2. **Downside Stability Factor (Low-Risk Quality Proxy):**
   Computed as the negative trailing 63-day realized downside semi-deviation over $[t-63, \, t)$:
   $$S_{\text{stab}, i, t} = -\sqrt{\frac{1}{63} \sum_{\tau=t-63}^{t-1} \min(r_{i, \tau}, 0)^2}$$

### 5.2 Cross-Sectional Peer Z-Score Normalization (#170)
Raw sub-factors are standardized cross-sectionally strictly across the active constituent cohort $\mathcal{A}_t = \{i : \text{mask}_{t,i} = \text{True}\}$ on date $t$:

$$z_{i, t} = \text{clip}\left( \frac{S_{i, t} - \mu_{\mathcal{A}_t}}{\sigma_{\mathcal{A}_t}}, \, -3.0, \, +3.0 \right)$$

* **Equal Blending & Re-Standardization:** The normalized sub-factors are combined ($0.5 \, z_{\text{drift}, i, t} + 0.5 \, z_{\text{stab}, i, t}$) and re-standardized across $\mathcal{A}_t$ so $q_t$ maintains zero cross-sectional mean and unit variance.
* **Inactive & Sparse Universe Guards:** Inactive constituents ($i \notin \mathcal{A}_t$) strictly receive $q_{i, t} = 0.0$. When $N_t < 2$ or $\sigma_{\mathcal{A}_t} < 10^{-6}$, $q_t = \mathbf{0}$ so the optimizer reverts cleanly to pure minimum variance.

### 5.3 Linear Quality Tilt Objective Function (#169)
The convex quadratic program integrates the quality vector via a linear reward term scaled by tilt intensity $\lambda \ge 0$:

$$\min_{w} \quad \frac{1}{2} w^{\top} \Sigma^{*} w - \lambda \, q_t^{\top} w$$

$$\text{subject to} \quad w \ge 0, \quad \sum_{i=1}^{N} w_i = B_t$$

Because $-\lambda \, q_t^{\top} w$ is affine in $w$ and $\Sigma^{*} \in \mathbb{S}_{+}^{N}$ is positive semi-definite, the composite objective remains strictly convex and Disciplined Convex Programming (DCP) compliant.

---

## 6. Concentration Boundary Constraints & Execution Architecture

### 6.1 Dynamic Upper Weight Cap ($w_{\max} = \min(0.35, \, 2 / N_t)$) (#174)
Unconstrained minimum-variance and quality-tilted optimizers tend to concentrate capital in a handful of low-volatility or high-momentum outliers—exposing the portfolio to single-issuer idiosyncratic event risk (such as executive transitions, earnings shocks, or distress in micro-cap names like `CARV` or `UONE`).

To enforce institutional diversification across changing universe sizes $N_t$, each constituent weight $w_i$ is bounded above by a dynamic regime-adaptive ceiling:

$$w_i \le w_{\max}(N_t) = \min\left(0.35, \, \frac{2}{N_t}\right)$$

| Active Universe Size ($N_t$) | Dynamic Cap $\min(0.35, 2/N_t)$ | Equity Budget ($B_t$) | Economic & Feasibility Rationale |
| :--- | :--- | :--- | :--- |
| **$N_t \ge 6$** | $\frac{2}{N_t} \le 33.33\%$ | $1.00$ ($100\%$) | Scales down proportionally as the roster expands (e.g., $11.76\%$ max at $N_t = 17$), preventing mega-cap dominance while allowing active $2\times$ overweight conviction relative to equal weight. |
| **$3 \le N_t \le 5$** | $35.00\%$ | $1.00$ ($100\%$) | Hard institutional ceiling prevents any single company from exceeding $35\%$ of capital while guaranteeing strict feasibility ($3 \times 0.35 = 1.05 > 1.00$). |
| **$N_t \in \{1, 2\}$** | $\min(0.35, 2/N_t) = 0.35$ | $\frac{1}{3}$ ($33.33\%$) | Sparse-universe concentration defense holds $\ge 66.67\%$ in cash; individual equity weights cannot exceed $33.33\% < 35\%$. |

### 6.2 Graceful Inverse-Volatility Fallback (#175)
If external custom bounds render the feasible set empty ($N_t \cdot w_{\max} < B_t$) or numerical ill-conditioning causes both `CLARABEL` and `OSQP` to return non-optimal statuses, the optimizer bypasses solver failure via a deterministic inverse-volatility fallback derived from the diagonal of $\Sigma^{*}$:

$$w_{i, \text{fallback}} = B_t \cdot \frac{\left(\max(\sqrt{\Sigma^{*}_{ii}}, \epsilon)\right)^{-1}}{\sum_{j \in \mathcal{A}_t} \left(\max(\sqrt{\Sigma^{*}_{jj}}, \epsilon)\right)^{-1}}$$

This preserves risk-parity proportionality and budget compliance without halting multi-decade rolling simulations.

### 6.3 Parameterized Problem Caching & Cholesky Reformulation (#176)
To eliminate repeated symbolic AST canonicalization across thousands of rolling rebalance dates, `ConvexPortfolioOptimizer` caches compiled `cvxpy` problem graphs keyed by `(n_assets, has_tilt, has_upper_bound)`. The quadratic variance form $w^{\top}\Sigma^{*}w$ is expressed via upper Cholesky factorization ($\Sigma^{*} = L^{\top}L$) as $\|L w\|_2^2$ (`cp.sum_squares(chol_param @ w)`), allowing rolling updates to mutate `cp.Parameter` values in place with `warm_start=True`.

---

## 7. Complete End-to-End Mathematical Specification (#184)

### 7.1 Rolling Point-in-Time Return Window & Causality
At trading session $t$, let $\mathcal{A}_t = \{i \in \{1, \dots, M\} : \text{mask}_{t,i} = 1\}$ denote the active Black CEO constituent set of cardinality $N_t = |\mathcal{A}_t|$, lagged by $t+1$ execution days. All statistical estimates use strictly historical arithmetic returns through session $t - 1$ over trailing lookback window $W = 63$ trading days:

$$R_{t-1}^{(W)} = \begin{bmatrix} r_{t-W, \mathcal{A}_t} \\ \vdots \\ r_{t-1, \mathcal{A}_t} \end{bmatrix} \in \mathbb{R}^{W \times N_t}$$

### 7.2 Ledoit-Wolf Shrinkage & Spectral PSD Conditioning
Let $\hat{\Sigma}_t = \frac{1}{W - 1} (R_{t-1}^{(W)} - \bar{R})^\top (R_{t-1}^{(W)} - \bar{R})$ be the sample covariance matrix and $\mu_t = \frac{\text{tr}(\hat{\Sigma}_t)}{N_t}$ be the isotropic target variance scale. The shrunk covariance matrix is:

$$\Sigma_{\text{LW}, t} = (1 - \delta_t)\hat{\Sigma}_t + \delta_t \mu_t I_{N_t}, \qquad \delta_t \in [0, 1]$$

To guarantee strict positive definiteness ($\Sigma_t^* \succ 0$) under rank-deficient or near-collinear regimes, spectral decomposition $\Sigma_{\text{LW}, t} = Q \Lambda Q^\top$ is floored at $\epsilon = 10^{-6}$:

$$\Sigma_t^* = Q \, \text{diag}\!\left(\max(\lambda_1, \epsilon), \dots, \max(\lambda_{N_t}, \epsilon)\right) Q^\top$$

### 7.3 Cross-Sectional Fundamental Quality Score ($q_t$)
For each active constituent $i \in \mathcal{A}_t$, raw momentum-to-downside-risk quality $s_{t,i}$ is computed from trailing mean drift $\bar{r}_{t,i}$ and downside semi-deviation $\sigma^-_{t,i} = \sqrt{\frac{1}{W}\sum_{k=1}^{W} \min(r_{t-k,i}, 0)^2}$:

$$s_{t,i} = \frac{\bar{r}_{t,i}}{\max(\sigma^-_{t,i}, \, \epsilon_{\text{vol}})}, \qquad \epsilon_{\text{vol}} = 10^{-4}$$

Scores are cross-sectionally standardized across active peers $\mathcal{A}_t$ to zero mean and unit variance:

$$q_{t,i} = \frac{s_{t,i} - \mu_{s,t}}{\max(\sigma_{s,t}, \, 10^{-8})}, \qquad i \in \mathcal{A}_t$$

### 7.4 Convex Quadratic Program & Cholesky Reformulation
Let $B_t = \min(1.0, \, N_t / 3)$ denote the sparse-universe equity budget and $w_{\max}(N_t) = \min(0.35, \, 2 / N_t)$ denote the dynamic single-asset ceiling. Factoring $\Sigma_t^* = L_t^\top L_t$ via upper Cholesky decomposition, the optimizer solves:

$$\begin{aligned}
\min_{w \in \mathbb{R}^{N_t}} \quad & \frac{1}{2} \|L_t w\|_2^2 - \lambda_{\text{tilt}} \, q_t^\top w \\
\text{subject to} \quad & \mathbf{1}^\top w = B_t \\
& 0 \le w_i \le w_{\max}(N_t), \qquad \forall i \in \{1, \dots, N_t\}
\end{aligned}$$

with cash allocation $w_{\text{cash}, t} = 1 - B_t$ earning the daily risk-free rate $r_{f,\text{daily}} = (1 + r_{f,\text{annual}})^{1/252} - 1$.

### 7.5 Synchronized Rebalancing Calendar & Friction Accounting
Target weights $w_t^{\text{target}}$ are recomputed on the synchronized rebalance indicator $\mathbb{I}_{\text{rebal}}(t) \in \{0, 1\}$ (initial session $t=0$, first trading day of each calendar month, or any universe entry/exit event). Between rebalances ($\mathbb{I}_{\text{rebal}}(t) = 0$), holdings drift proportionally with realized returns while preserving $B_t$.

One-way portfolio turnover $\tau_t$ and institutional transaction friction $c_t$ (at $\kappa = 10\text{ bps} = 0.0010$) are deducted daily:

$$\tau_t = \frac{1}{2} \left( \sum_{i=1}^{M} \left|w_{t,i}^{\text{target}} - w_{t-1,i}^{\text{drift}}\right| + \left|w_{\text{cash}, t}^{\text{target}} - w_{\text{cash}, t-1}^{\text{drift}}\right| \right), \qquad c_t = \kappa \cdot \tau_t$$

$$r_{\text{net}, t} = \left(\sum_{i=1}^{M} w_{t,i}^{\text{target}} r_{t,i} + w_{\text{cash}, t}^{\text{target}} r_{f,\text{daily}}\right) - c_t$$