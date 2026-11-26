# DSC@NYU Project Expo 2026 — 5-Slide Presentation Deck (`BCEOS <Index>`)

**Presenter:** Tanesha Fuller (Solo Researcher)  
**Track:** Quantitative Finance & Applied Econometrics  
**Live Interactive Terminal:** https://dsc-project-expo2026-tef3573.vercel.app  

---

## Slide 1: Title, Author & Research Motivation
* **Title:** *The Black Representation in Alpha (`BCEOS <Index>`): A Point-in-Time Equity Backtest, Convex Optimization & Multi-Factor Attribution (2000–2026)*
* **Plain-English Concept (The "Hypothetical ETF"):** Imagine a rules-based index fund that only holds publicly traded U.S. companies while actively led by a Black CEO—buying the trading day after appointment and selling the trading day after departure.
* **Research Motivation & Empirical Gap:** Prior corporate governance studies often suffer from **survivorship bias** (dropping acquired or bankrupt firms) and **look-ahead bias**. Over 27 years (`2000-01-03` to `2026-11-24`, `6,765` NYSE sessions across `40` CEOs and `36` issuers), we test whether this leadership screen outperforms the S&P 500 (`SPY` and equal-weight `RSP`) under institutional execution controls.
* **Headline Result:** **12.83% Net CAGR** (`25.55x` wealth multiple) vs. **8.01% SPY** (`+482 bps/yr` net of `10 bps` friction), generating **+10.00% annualized Jensen's Alpha** (`+7.90%` vs. `RSP`).

---

## Slide 2: Survivorship-Bias-Free Methodology & Point-in-Time Masking
* **Dynamic Membership Matrix (`src/universe.py`):** Maps exact executive start and end timestamps across `36` U.S. public tickers (`6,765` sessions, active breadth $N_t \in [0, 17]$, mean $9.43$).
* **Zero Survivorship Bias (`src/data_loader.py`):** Integrates local historical archives for delisted and acquired firms (`AET`, `GBT`, `JCP`, `PBPB`, `WBA`) and stitches corporate restructurings (`QVCG` $\rightarrow$ `QRTEA`).
* **Strict $t+1$ Causal Execution Lag:** Shifts executive transition flags forward by one trading day ($I_{i,t} = I_{i,t-1}^{\text{raw}}$) so portfolio weights never condition on same-day announcements.
* **Early-Era Cash Guardrail ($N_t < 3$) & Realistic Friction (`src/backtest.py`):** Caps equity exposure at $B_t = \min(1, N_t/3)$ when fewer than 3 CEOs are active (parking residual capital in cash), models daily weight drift, and deducts `10 bps` per unit of one-way turnover (`73.24%` annualized turnover; `1.97%` total cumulative friction).

---

## Slide 3: CVXPY Convex Optimization & Shrinkage Covariance
* **Three Allocation Schemes (`src/optimizer.py`):** Compares Equal-Weight (`1/N_t`), Inverse-Volatility (`1/\sigma_i`), and a regularized **Convex-Quality Quadratic Program (`QP`)** solved via `cvxpy` (`CLARABEL` / `OSQP` / `SCS`).
* **Shrinkage Covariance Conditioning ($W = 63\text{d}$):** Regularizes trailing sample covariance via $\Sigma^*_t = (1 - \lambda)\hat{\Sigma}_t + \lambda\,\mathrm{diag}(\hat{\Sigma}_t) + \epsilon I$ ($\lambda_{\min} \ge \epsilon > 0$) to ensure strict positive semi-definiteness even in small-$N$ regimes.
* **Quality Tilt & Concentration Controls:** Minimizes $\frac{1}{2} w^{\top}\Sigma^*_t w - \gamma q_t^{\top} w + \eta \|w - w_{\text{prev}}\|_2^2$ where $q_t$ is the cross-sectional $z$-score of trailing return drift over downside semi-deviation, subject to $\sum w_i = B_t$ and a hard **$35\%$ single-issuer ceiling**.
* **Dynamic Econometric Risk Overlay (`src/timeseries.py`):** Pairs allocation with `GARCH(1,1)` / `GJR-GARCH(1,1)` conditional volatility modeling (`0.9983` full-horizon persistence, compressing to `0.9061` and a `7.0d` shock half-life post-2010).

---

## Slide 4: Empirical Results — Scorecard & Benchmark Outperformance
* **Full-Horizon Cumulative Compounding (`2000–2026`, Net of `10 bps` Friction):**
  * **BCEOS Equal-Weight (Net):** `12.83%` CAGR | `25.55x` Wealth (`$10k` $\rightarrow$ `$255.5k`) | `0.44` Sharpe | `0.83` Sortino | `0.86` Beta vs. `SPY`
  * **S&P 500 Cap-Weighted (`SPY`):** `8.01%` CAGR | `19.27%` Ann. Vol | `0.42` Sharpe | `55.19%` MDD
  * **S&P 500 Equal-Weighted (`RSP`, post-May 2003):** `11.15%` CAGR | `19.69%` Ann. Vol | `0.56` Sharpe | `59.92%` MDD
* **Structural Maturation (Pre-2010 vs. Post-2010 Split):** As active breadth expanded from $N_t \le 7$ to $15\text{–}17$ concurrent CEOs, post-2010 CAGR rose to **`16.36%`** (`1.00` Sortino, `47.47%` MDD, `7.0d` `GARCH` half-life) vs. `13.66%` for `SPY`.
* **Macro-Regime Resilience (`src/macro_regimes.py`):** Outperformed `SPY` in the Dot-Com Bust (`+8.16%` vs. `-42.64%`, `0.62x` downside capture), ZIRP/QE Era (`+960.47%` vs. `+520.77%`), COVID-19 Surge (`+141.48%` vs. `+44.91%`), and AI/High-Rate Regime (`+92.86%` vs. `+74.17%`, `17.65%` MDD).

---

## Slide 5: Fama-French Factor Attribution, Tech Stack & Policy Takeaways
* **Fama-French 5-Factor (`FF5`) OLS with Newey-West HAC (`src/attribution.py`):** Controlling for market (`Mkt-RF`), size (`SMB`), value (`HML`), profitability (`RMW`), and investment (`CMA`) confirms **+10.00% annualized Jensen's Alpha** vs. `SPY` (`+7.90%` vs. `RSP`) with heteroskedasticity- and autocorrelation-consistent standard errors.
* **Non-Parametric Bootstrap & Weak-Form Efficiency (`src/bootstrap.py`, `src/arima_diagnostics.py`):** Politis-Romano stationary block bootstrap validates tail-aware confidence intervals, while `6,513`-session walk-forward `ARIMA(1,0,1)` (`OOS R2 = -0.0774%`) and Ljung-Box Lag-1 ($p = 0.1741$) rule out stale-price drift.
* **Full-Stack Architecture & Live Workstation:** Python 3.11 (`NumPy`, `Pandas`, `SciPy`, `CVXPY`, `statsmodels`, `arch`), `Plotly.js` 7-screen Bloomberg-style terminal (`WEI`, `DES`, `MEMB`, `WEALTH`, `ECST`, `VOL`, `ALPH`), Vercel serverless `/api/live_market`, `87` unit tests, Docker, and automated GitHub Actions daily refresh.
* **Policy & Governance Takeaways:** Executive representation barriers act as a rigorous selection filter rather than a performance drag; expanding C-suite breadth compresses idiosyncratic risk while delivering durable long-horizon compounding.
