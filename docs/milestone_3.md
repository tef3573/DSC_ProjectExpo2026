# Milestone 3: Convex Quadratic Programming Optimizer & Risk-Controlled Allocation

## 1. Objective & Quantitative Mandate
While the Milestone 2 equal-weighted baseline achieved strong cumulative growth ($13.36\%$ CAGR), equal weighting concentrated risk in volatile small- and mid-cap names, producing $42.09\%$ annualized volatility and a $72.69\%$ maximum drawdown. Milestone 3 introduces a regularized convex quadratic programming (QP) allocation engine (`src/optimizer.py`) that compresses portfolio variance, tilts toward high-quality momentum/low-downside-risk constituents, enforces a strict $35\%$ single-name concentration ceiling, and eliminates intra-month churn via synchronized monthly rebalancing.

---

## 2. Core Modules & Data Artifacts

| Artifact / Module | Role in Pipeline | Key Capabilities & Verification Coverage |
| :--- | :--- | :--- |
| `src/optimizer.py` | Portfolio weighting & convex QP solver module | Implements Equal-Weight, Inverse-Volatility, PSD covariance estimation, quality $z$-scores, and cached `cvxpy` QP solver |
| `docs/optimizer.md` | Mathematical formulation & solver specification | Documents shrinkage covariance conditioning, quality tilt objective, and concentration feasibility rules |
| `data/comparative_schemes.csv` | 3-scheme comparative daily return & wealth dataset | Exports aligned net return curves for `Equal_Weight_Net`, `Inverse_Vol_Net`, and `Convex_Quality_Net` |
| `tests/test_optimizer.py` | Comprehensive optimizer & stress-window test suite | Verifies PSD eigenvalues ($\lambda_{\min} \ge \epsilon$), $35\%$ cap, cached `cp.Problem` reuse, and historical rebalance feasibility |

---

## 3. Mathematical Formulation & Risk Controls

### 3.1 Trailing Covariance Conditioning ($W = 63$ Trading Days)
To guarantee strict convexity and positive semi-definiteness even under collinear asset returns or sparse cross-sections, trailing sample covariance $\hat{\Sigma}_t$ over $[t - W, t - 1]$ ($W = 63$ sessions) is regularized via diagonal shrinkage and eigenvalue flooring:

$$
\Sigma^*_t = (1 - \lambda)\hat{\Sigma}_t + \lambda\,\mathrm{diag}(\hat{\Sigma}_t) + \epsilon I_N, \qquad \lambda_{\min}(\Sigma^*_t) \ge \epsilon > 0
$$

### 3.2 Cross-Sectional Quality Tilt Vector ($q_t$)
For active constituents ($I_{i, t} = 1$), composite quality combines trailing 63-day cumulative return drift $\mu_{i, t}$ with trailing downside volatility $\sigma_{i, t}^{-}$:

$$
\text{raw}_{i, t} = \frac{\mu_{i, t}}{\max(\sigma_{i, t}^{-},\; \epsilon_{\sigma})}, \qquad q_{i, t} = \frac{\text{raw}_{i, t} - \bar{\text{raw}}_t}{\text{std}(\text{raw}_t)} \quad (\text{and } q_{i, t} = 0 \text{ for } I_{i, t} = 0)
$$

### 3.3 Regularized Convex QP Objective & Concentration Constraints
At each synchronized monthly rebalance date (plus any session where active universe membership $I_t$ changes), target weights $w \in \mathbb{R}^N$ are solved via `cvxpy` (`CLARABEL` / `OSQP` / `SCS` fallback chain with dimension-keyed parameterized `cp.Problem` caching):

$$
\min_{w \in \mathbb{R}^N} \; \frac{1}{2} w^{\top} \Sigma^*_t w - \gamma\, q_t^{\top} w + \eta \|w - w_{\text{prev}}\|_2^2
$$

subject to linear budget, long-only, and dynamic single-name concentration constraints:

$$
\sum_{i=1}^{N} w_i = B_t = \min\left(1.0,\; \frac{N_t}{3}\right), \qquad 0 \le w_i \le u_i \cdot I_{i, t}, \qquad u_i = \min\left(0.35,\; \max\left(0.35,\; \frac{1.5 B_t}{N_t}\right)\right)
$$

* **Sparse-Universe Protection:** When $N_t < 3$, $B_t = N_t / 3 \le 2/3$, ensuring every active weight satisfies $w_i \le 0.35$ strictly.
* **Infeasible Cap Safeguard:** If $N_t \cdot 0.35 < B_t$ under custom overrides, the solver falls back deterministically to bounded inverse-volatility weights.

### 3.4 Synchronized Monthly Rebalancing & Turnover Control
* **Rebalance Trigger:** Portfolio weights rebalance strictly on the **first trading session of each calendar month** or when active constituent membership changes ($\Delta I_t \ne 0$). Between monthly rebalances, weights drift naturally without incurring daily rebalancing turnover.

---

## 4. Three-Scheme Comparative Evaluation & Stress Testing
* **Schemes Evaluated (`run_comparative_backtests`):**
  1. `Equal-Weight` (`Black_CEO_Net`) — Unoptimized equal-weight benchmark across active Black CEO firms.
  2. `Inverse-Volatility` (`Inverse_Vol_Net`) — Trailing volatility-weighted risk-parity baseline.
  3. `Convex-Quality` (`Convex_Quality_Net`) — Regularized minimum-variance QP with quality tilts, $35\%$ cap, and monthly rebalancing.
* **Historical Crisis & Synthetic Stress Verification:**
  * Verified finite weights, exact budget unity ($\sum w_i + w_{\text{cash}} = 1.0$), and volatility compression across both the **2008 Global Financial Crisis** and **2020 COVID-19 Crash** windows.
  * Confirmed solver stability under synthetic $5\times$ volatility shocks, near-collinear correlation spikes ($\rho \to 0.999$), and sudden universe contractions ($N_t: 6 \rightarrow 2$).
