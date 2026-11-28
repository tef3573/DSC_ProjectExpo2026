# Quantitative Methodology & Econometric Specification (`BCEOS <Index>`)

**Evaluation Window:** `2000-01-03` to `2026-11-25` (6,766 NYSE trading sessions)  
**Risk-Free Rate Benchmark ($R_f$):** 1.50% annualized  
**Minimum Diversification Threshold ($N_{\min}$):** 3 active issuers  
**Transaction Friction ($\tau$):** 10.0 bps proportional round-trip turnover cost  

---

## 1. Universe Construction & Anti-Bias Controls

To evaluate whether active Black executive leadership at publicly traded U.S. firms generates persistent risk-adjusted excess returns, we construct a survivorship-bias-free constituent universe spanning **40 Black CEO tenures across 36 unique public issuers** from January 2000 through November 2026.

1. **Elimination of Survivorship Bias:** Historical constituents that were acquired, taken private, or delisted during or subsequent to a Black CEO's tenure—including **Aetna (`AET`)**, **Global Blood Therapeutics (`GBT`)**, **J.C. Penney (`JCP`)**, **Qurate Retail (`QRTEA`)**, **Potbelly (`PBPB`)**, and **Walgreens Boots Alliance (`WBA`)**—are reconstructed using total-return price histories through their final public trading date rather than omitted retrospectively.
2. **Strict $t+1$ Execution Lag (Zero Look-Ahead Bias):** For any issuer $i$ whose CEO tenure begins on trading day $t_0$ and concludes on trading day $t_1$, the binary eligibility indicator $I_{i,t} \in \{0, 1\}$ is lagged by one full NYSE session:
   $$I_{i,t} = \mathbb{I}(t_0 + 1 \le t \le t_1 + 1)$$
   Portfolio entries execute at the closing auction on the trading day *following* public appointment, and exits execute at the closing auction on the trading day *following* departure.

---

## 2. Portfolio Accounting, Cash Drag & Transaction Friction

Let $N_t = \sum_{i=1}^{M} I_{i,t}$ denote the number of active eligible issuers on trading session $t$. During early sample years when representation was sparse ($N_t < 3$), full capital deployment would induce extreme single-issuer idiosyncratic concentration.

* **Sparse-Universe Cash Buffer ($B_t$):** Active equity exposure is scaled linearly below the diversification floor $N_{\min} = 3$:
  $$B_t = \min\!\left(1.0,\; \frac{N_t}{3}\right), \qquad w_{\text{cash},t} = 1.0 - B_t$$
  Uninvested capital earns the daily risk-free accrual $r_{f,t} = R_f / 252$.
* **Daily Drift & Proportional Turnover Cost:** Between rebalance events, target weights $w_{i,t}^*$ drift organically with realized daily total returns $r_{i,t}$:
  $$\tilde{w}_{i,t} = \frac{w_{i,t-1} (1 + r_{i,t-1})}{1 + R_{p,t-1}^{\text{gross}}}$$
  On any session where target weights adjust, one-way portfolio turnover is $\text{TO}_t = \frac{1}{2} \sum_i |w_{i,t}^* - \tilde{w}_{i,t}|$ (plus cash weight delta), and net strategy return deducts $\tau = 10\text{ bps}$ ($0.0010$):
  $$R_{p,t}^{\text{net}} = \sum_{i=1}^{M} w_{i,t}^* r_{i,t} + w_{\text{cash},t} r_{f,t} - \tau \cdot \text{TO}_t$$

---

## 3. Weighting Schemes & Convex Quadratic Optimization

We benchmark three systematic portfolio construction rules under identical $t+1$ execution and friction constraints:

1. **Equal-Weight (`BCEOS EW`):** Assigns $w_{i,t}^* = B_t / N_t$ across all active constituents ($I_{i,t} = 1$), isolating pure leadership breadth without market-cap concentration.
2. **Inverse-Volatility (`BCEOS Inv-Vol`):** Weights constituents inversely proportional to trailing 63-day realized return volatility $\hat{\sigma}_{i,t}$, scaled to sum to $B_t$.
3. **Convex Quality-Tilted Quadratic Program (`BCEOS Convex QP`):** Solves a strictly convex quadratic program at each rebalance via `CVXPY` using the Ledoit-Wolf shrinkage covariance estimator $\hat{\Sigma}_t^{\text{LW}}$ and standardized fundamental quality scores $q_i$:
   $$\max_{w \in \mathbb{R}^M} \; \lambda_q q^\top w - \frac{\gamma}{2} w^\top \hat{\Sigma}_t^{\text{LW}} w \quad \text{s.t.} \quad \sum_{i=1}^M w_i = B_t, \quad 0 \le w_i \le 0.35 \cdot I_{i,t}$$
   where the $35\%$ upper bound prevents single-issuer dominance while Ledoit-Wolf shrinkage guarantees positive semi-definiteness when $N_t$ is small.

---

## 4. Econometric Inference & Volatility Regime Modeling

To distinguish genuine leadership-linked excess returns from passive factor exposures or small-sample luck, we apply four complementary econometric diagnostics:

1. **CAPM & Fama-French 5-Factor Regression with Newey-West HAC Errors:**
   $$R_{p,t}^{\text{net}} - r_{f,t} = \alpha + \beta_{\text{MKT}}(\text{Mkt-RF}_t) + \beta_{\text{SMB}}\text{SMB}_t + \beta_{\text{HML}}\text{HML}_t + \beta_{\text{RMW}}\text{RMW}_t + \beta_{\text{CMA}}\text{CMA}_t + \varepsilon_t$$
   Standard errors and $t$-statistics are computed using the Newey-West heteroskedasticity- and autocorrelation-consistent (HAC) covariance estimator to account for volatility clustering and serial correlation in daily equity returns.
2. **Politis-Romano Stationary Block Bootstrap:** To evaluate finite-sample alpha significance without Gaussian normality assumptions, we resample daily return blocks with geometrically distributed block lengths (preserving weak time-series dependence) across 10,000 Monte Carlo replications.
3. **Conditional Volatility & Regime Persistence (`GARCH(1,1)`):** We estimate a $\text{GARCH}(1,1)$ model $\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$ across both the full 2000–2026 sample and the post-2010 maturation regime ($N_t \ge 10$), quantifying volatility persistence $\hat{\alpha}_1 + \hat{\beta}_1$ and shock half-life $h = \ln(0.5) / \ln(\hat{\alpha}_1 + \hat{\beta}_1)$.
4. **Return Serial Correlation (`ARIMA` / Ljung-Box):** Conditional mean dynamics are audited via `ARIMA` order selection and Ljung-Box $Q$-tests to verify residual whiteness after factor adjustment.

---

## 5. Audited Full-Horizon Scorecard (`2000-01-03` – `2026-11-25`)

| Performance Metric | `BCEOS` Equal-Weight (Net) | `SPY` (Cap-Weighted) | `RSP` (Equal-Weight, Post-2003*) |
| :--- | :---: | :---: | :---: |
| **Annualized CAGR** | **12.88%** (Gross: 12.97%) | 7.99% | 11.19% |
| **Cumulative Wealth Multiple** | **25.89x** (Gross: 26.41x) | 7.88x | 12.14x |
| **Annualized Volatility** | 38.33% | 19.27% | 19.69% |
| **Sharpe Ratio ($R_f = 1.5\%$)** | **0.44** | 0.42 | 0.56 |
| **Sortino Ratio** | **0.83** | 0.69 | 0.81 |
| **Calmar Ratio** | 0.18 | — | — |
| **Maximum Drawdown (MDD)** | 70.44% | 55.19% | 59.92% |
| **CAPM Beta ($\beta$)** | 0.86 | 1.00 | 0.90 |
| **Annualized Jensen's Alpha ($\alpha$)** | **+10.05%** | 0.00% | +7.92% (vs. RSP) |
| **Annualized Turnover / Friction Drag** | 73.23% / 1.97% | Passive | Passive |

*\*Note: `RSP` (Invesco S&P 500 Equal Weight ETF) is evaluated over its exact post-inception window (`2003-05-02` to `2026-11-25`, 5,931 sessions), where `BCEOS` Net CAGR is 13.41%.*

---

## 6. Academic References

1. **Bollerslev, T. (1986).** Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
2. **Fama, E. F., & French, K. R. (2015).** A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1–22.
3. **Jensen, M. C. (1968).** The Performance of Mutual Funds in the Period 1945–1964. *Journal of Finance*, 23(2), 389–416.
4. **Ledoit, O., & Wolf, M. (2004).** A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
5. **Newey, W. K., & West, K. D. (1987).** A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.
6. **Politis, D. N., & Romano, J. P. (1994).** The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
7. **Sortino, F. A., & Price, L. N. (1994).** Performance Measurement in a Downside Risk Framework. *Journal of Investing*, 3(3), 59–64.
