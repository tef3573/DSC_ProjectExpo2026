# Executive Leadership & Risk-Adjusted Equity Performance
**Hypothetical Black CEO Equity Index (`BCEOS <Index>`) | 2000–2026 Empirical Backtest**  
**Author:** Tanesha | NYU Data Science Club Project Expo 2026  

---

## Slide 1: Title, Author & Research Motivation
- **Research Question:** Do publicly traded U.S. companies led by Black Chief Executive Officers generate statistically significant risk-adjusted alpha over a multi-decade horizon?
- **The Empirical Gap:** Prior corporate governance studies often suffer from **survivorship bias** (dropping acquired/delisted firms like Aetna, GBT, JCPenney, and Walgreens) and **look-ahead bias** (trading on announcement dates before public disclosure).
- **Contribution:** We construct `BCEOS <Index>`, a point-in-time, survivorship-bias-free daily equity index spanning **40 Black CEO tenures across 36 U.S. public issuers** from **January 3, 2000 to November 23, 2026 (6,764 trading sessions)**.
- **Speaker Notes (45s):** Introduce the motivation behind treating executive leadership representation as an institutional factor/index design problem, emphasizing strict point-in-time engineering over retrospective anecdotes.

---

## Slide 2: Survivorship-Bias-Free Methodology & Point-in-Time Masking
- **Point-in-Time Universe Construction (`src/universe.py`):**
  - Tracks exact SEC Form 8-K appointment and departure windows across **36 issuers**, including **6 delisted/acquired tickers** (`AET`, `GBT`, `JCP`, `QRTEA`, `PBPB`, `WBA`) reconstructed from historical CRSP/archive series.
  - Active breadth evolves organically from $N_t = 4$ firms in 2000 to $N_t = 17$ firms in 2026.
- **Strict $t+1$ Causal Execution Lag & Friction Model (`src/backtest.py`):**
  - Portfolio membership shifts take effect on the **next trading day ($t+1$)** to eliminate intraday look-ahead bias.
  - Dynamic **33.3% single-issuer concentration cap** (allocating residual weight to risk-free cash when $N_t < 3$).
  - Realistic **10 bps one-way transaction cost** deducted on every rebalance and turnover event.
- **Speaker Notes (60s):** Walk through how delisted mergers (e.g., Aetna's acquisition by CVS, GBT's acquisition by Pfizer) and $t+1$ lagging guarantee an implementable, real-world trading simulation.

---

## Slide 3: CVXPY Convex Optimization & Shrinkage Covariance
- **Two Portfolio Weighting Schemes:**
  1. **Equal-Weight (`Black_CEO_Net`):** Benchmark $1/N_t$ allocation capturing pure breadth across all active Black CEO issuers.
  2. **Convex-Quality QP (`Convex_Quality_Net`):** Solves a daily convex quadratic program via `CVXPY` minimizing portfolio variance subject to a **35% max position cap** and full investment constraints.
- **Ledoit-Wolf Covariance Shrinkage (`src/optimizer.py`):**
  - Regularizes sample covariance matrices over trailing estimation windows to prevent extreme weight concentration when $N_t$ is small.
- **Dynamic GARCH(1,1) Volatility-Targeting Overlay (`src/timeseries.py`):**
  - Walk-forward 1-step variance forecasts scale equity exposure $w_t \in [0\%, 100\%]$ to target **20.0% annualized volatility** (matching SPY's 19.27% risk profile).
- **Speaker Notes (60s):** Explain how convex optimization and GARCH variance targeting address the concentration risk of early-2000s small-$N_t$ cohorts.

---

## Slide 4: Empirical Results — Institutional Scorecard & Benchmark Outperformance
| Performance Metric (2000–2026) | BCEOS Eq-Wt (Net) | Convex-Quality QP (Net) | Causal 20% Vol-Target | S&P 500 (`SPY`) | S&P Eq-Wt (`RSP`*) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Net CAGR** | **12.74%** | **9.65%** | **9.52%** | 8.03% | 11.13% |
| **Cumulative Wealth ($10k Base)** | **$249,800 (24.98x)** | **$117,400 (11.74x)** | **$113,800 (11.38x)** | $79,400 (7.94x) | $120,200 (12.02x) |
| **Annualized Volatility** | 38.33% | 24.81% | **19.93%** | 19.27% | 19.69% |
| **Sharpe / Sortino Ratio** | **0.44 / 0.83** | 0.44 / 0.72 | **0.48 / 0.85** | 0.42 / 0.58 | 0.56 / 0.74 |
| **Maximum Drawdown (MDD)** | -70.44% | -59.82% | **-51.11%** | -55.19% | -59.92% |
| **Post-2010 GARCH(1,1) Half-Life** | **7.0 days** | 365.7 days | **7.0 days** | 20.8 days | — |

- **Key Insights:**
  - Outpaces cap-weighted `SPY` by **+471 bps/yr** net of **73.25% annualized turnover** (only **1.97%** cumulative friction drag over 27 years).
  - Post-2010 breadth expansion ($N_t 	o 17$) accelerated net CAGR to **16.21%** and cut volatility shock half-life by **95%** (from 146.8d to **7.0d**).
- **Speaker Notes (60s):** Highlight the Sortino ratio (0.83 vs. 0.44 Sharpe), showing that volatility is heavily skewed toward upside breakouts rather than permanent capital loss.

---

## Slide 5: Fama-French 5-Factor Attribution, Monte Carlo & Takeaways
- **Fama-French 5-Factor Newey-West HAC Regression (`src/attribution.py`):**
  - Controls for Market (`Mkt-RF` $eta = 0.86$), Size (`SMB` $= +0.42$), Value (`HML` $= +0.29$), Profitability (`RMW` $= -0.18$), and Investment (`CMA` $= -0.11$).
  - **Annualized FF5 Alpha:** **+9.90% to +10.52%** ($t > 2.5, p < 0.01$), confirming idiosyncratic alpha beyond small-cap or value factor tilts.
- **10,000-Trial Monte Carlo Permutation Bootstrap (`src/bootstrap.py`):**
  - Outperforms **97.6%** of 10,000 randomly sampled size-matched S&P peer cohorts ($p = 0.024$).
- **Martingale Market Efficiency (`src/arima_diagnostics.py`):**
  - Walk-forward `ARIMA(1,0,1)` out-of-sample $R^2 = -0.055\%$ and Ljung-Box Lag-1 $p = 0.1164$ rule out stale-price autocorrelation.
- **Live Interactive Workstation (`1-BLOOMBERG`):** Built a 7-screen institutional terminal (`WEI`, `DES`, `MEMB`, `GP`, `ECST`, `VOL`, `ALPH`) with automated 5:30 PM ET post-close market updates.
- **Speaker Notes (45s):** Conclude with robustness takeaways and transition to the live `1-BLOOMBERG` terminal demonstration.
