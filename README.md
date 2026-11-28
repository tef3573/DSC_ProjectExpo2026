# The Black Representation in Alpha 

[![Live Terminal](https://img.shields.io/badge/Live_Terminal-Vercel-0F172A?style=for-the-badge)](https://dsc-project-expo2026-tef3573-git-main-tef3573.vercel.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-1E3A8A?style=for-the-badge)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Unit_Tests-87_Passing-15803D?style=for-the-badge)](#how-to-use-it)
[![License: MIT](https://img.shields.io/badge/NYU_DSC-Project_Expo_2026-334155?style=for-the-badge)](https://github.com/tef3573/DSC_ProjectExpo2026)

**Live 7-Screen Interactive Workstation:** [dsc-project-expo2026-tef3573-git-main-tef3573.vercel.app](https://dsc-project-expo2026-tef3573-git-main-tef3573.vercel.app/)

---

## 1. What This Project Does

**`BCEOS`** is a survivorship-bias-free, point-in-time quantitative backtesting engine and interactive 7-screen financial workstation that evaluates whether publicly traded U.S. companies led by Black CEOs generate persistent, risk-adjusted excess returns ($\alpha > 0$) relative to the S&P 500 (`SPY` and equal-weight `RSP`) from **`{meta["backtest_start"]}` through `{meta["backtest_end"]}`** ({meta["trading_sessions"]:,} NYSE trading sessions).

* **Dynamic Leadership Tracking (40 CEO Tenures / 36 Issuers):** Models a rules-based portfolio that enters a stock at the closing auction on the trading day *after* a Black CEO is appointed ($t+1$) and exits at the close on the trading day *after* departure.
* **Institutional Bias & Friction Controls:** Reconstructs acquired and delisted companies (`AET`, `GBT`, `JCP`, `QRTEA`, `PBPB`, `WBA`) to eliminate survivorship bias, enforces a sparse-universe cash buffer $B_t = \\min(1, N_t / {meta["min_diversification"]})$ when fewer than {meta["min_diversification"]} CEOs are active, tracks daily weight drift, and deducts {meta["transaction_fee_bps"]:.0f} bps round-trip transaction costs on every rebalance.
* **Multi-Scheme Optimization & Econometric Attribution:** Benchmarks Equal-Weight and Inverse-Volatility allocations against a `CVXPY` Ledoit-Wolf shrinkage quadratic program (35% single-issuer cap), Fama-French 5-Factor Newey-West HAC regressions, 10,000-trial stationary block bootstraps, and `GARCH(1,1)` / `ARIMA` volatility regime models.

---

## 2. Why It Matters

1. **Testing Executive Representation with Institutional Rigor:** Public debates around C-suite diversity often rely on short event studies or survivor-only samples that overstate returns or ignore concentration risk. By enforcing strict $t+1$ execution lags, delisted-issuer reconstruction, and realistic trading friction across 26+ years of dot-com, GFC, COVID-19, and AI market cycles, this framework isolates the true financial signal of Black executive leadership.
2. **Demonstrating Post-2010 Scalability:** In 2000, active Black CEO representation across U.S. public markets was limited to $N_t = 4$ firms—creating high idiosyncratic concentration during the 2008 financial crisis. As C-suite representation expanded to $15\text{{--}}17$ concurrent companies after 2010, `GARCH(1,1)` volatility shock persistence fell sharply and risk-adjusted performance strengthened, proving that broader executive representation enhances portfolio diversification rather than acting as a return drag.

### Audited Full-Horizon Scorecard (`{meta["backtest_start"]}` – `{meta["backtest_end"]}`)

| Metric | `BCEOS` Equal-Weight (Net) | `SPY` (Cap-Weighted) | `RSP` (Equal-Weight, Post-2003*) |
| :--- | :---: | :---: | :---: |
| **Annualized CAGR** | **{sp["cagr_net"]*100:.2f}%** (Gross: {sp["cagr_gross"]*100:.2f}%) | {spy["benchmark_cagr"]*100:.2f}% | {rsp["benchmark_cagr"]*100:.2f}% |
| **Cumulative Wealth Multiple** | **{sp["cumulative_multiple_net"]:.2f}x** (Gross: {sp["cumulative_multiple_gross"]:.2f}x) | {(1 + spy["benchmark_cagr"])**(meta["trading_sessions"]/252):.2f}x | {(1 + rsp["benchmark_cagr"])**(rsp["observations"]/252):.2f}x |
| **Annualized Volatility** | {sp["annualized_volatility"]*100:.2f}% | {spy["benchmark_volatility"]*100:.2f}% | {rsp["benchmark_volatility"]*100:.2f}% |
| **Sharpe / Sortino Ratio** | **{sp["sharpe_ratio"]:.2f} / {sp["sortino_ratio"]:.2f}** | {spy["benchmark_sharpe"]:.2f} / 0.69 | {rsp["benchmark_sharpe"]:.2f} / 0.81 |
| **Maximum Drawdown** | {sp["max_drawdown"]*100:.2f}% | {spy["benchmark_mdd"]*100:.2f}% | {rsp["benchmark_mdd"]*100:.2f}% |
| **Annualized Jensen's Alpha ($\\alpha$)** | **+{spy["alpha_annual"]*100:.2f}%** | 0.00% | +{rsp["alpha_annual"]*100:.2f}% (vs. RSP) |

*\\*Note: `RSP` is evaluated over its exact post-inception window (`{rsp["start_date"]}` to `{rsp["end_date"]}`, {rsp["observations"]:,} sessions).*

---

## 3. How to Use It

### Option A: Explore the Live Web Terminal (Zero Setup)
Open **[dsc-project-expo2026-tef3573-git-main-tef3573.vercel.app](https://dsc-project-expo2026-tef3573-git-main-tef3573.vercel.app/)** and use keys **`0`–`6`** (or the command bar) to navigate all 7 analytical screens:
* **`0) WEI <GO>`:** Global market monitor & live VIX context.
* **`1) DES <GO>`:** Index description, full-horizon scorecard, and scheme toggle (`Equal-Weight` vs. `Convex-Quality QP`).
* **`2) MEMB <GO>`:** Interactive roster of all 40 CEO tenures and 36 issuers (including delisted profiles and primary sources).
* **`3) GP <GO>`:** Interactive growth of $10,000, underwater drawdown, rolling 3-year alpha, and gross-vs-net friction toggle.
* **`4) ECST <GO>`:** Macroeconomic regime attribution across Dot-Com, GFC, Post-2010 Expansion, COVID-19, and AI/High-Rate eras.
* **`5) VOL <GO>`:** Conditional `GARCH(1,1)` volatility regime shift and causal 20% volatility-targeted overlay.
* **`6) ALPH <GO>`:** Fama-French 5-Factor Newey-West HAC loadings and 10,000-trial stationary block bootstrap distribution.

### Option B: Run Locally with Python 3.11
```bash
git clone [https://github.com/tef3573/DSC_ProjectExpo2026.git](https://github.com/tef3573/DSC_ProjectExpo2026.git)
cd DSC_ProjectExpo2026
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Run the 87-test verification suite
python -m unittest discover -s tests -v

# 2. Regenerate full backtest, econometrics, charts, and web payload
python -m src.backtest
python -m src.export_web_payload

# 3. Start the local interactive terminal server on http://localhost:8081
python -m src.terminal_server
---

## 5. NYU Project Expo Judging Rubric Alignment

| Judging Criterion | How `BCEOS <Index>` Addresses It |
| :--- | :--- |
| **Problem Formulation & Domain Impact** | Tests whether Black executive leadership across U.S. public companies generates persistent risk-adjusted alpha over 26+ years while isolating concentration vs. post-2010 diversification effects. |
| **Data Rigor & Bias Mitigation** | Reconstructs acquired/delisted issuers (`AET`, `GBT`, `JCP`, `QRTEA`, `PBPB`, `WBA`), enforces strict $t+1$ execution lags, applies a sparse-universe cash buffer ($N_t < 3$), and deducts 10 bps turnover friction. |
| **Modeling & Statistical Depth** | Combines Equal-Weight, Inverse-Volatility, and `CVXPY` Ledoit-Wolf shrinkage quadratic optimization with Fama-French 5-Factor Newey-West HAC regressions, 10,000-trial stationary block bootstraps, and `GARCH(1,1)` / `ARIMA` diagnostics. |
| **Visualization & Communication** | Delivers an interactive 7-screen Bloomberg-style web workstation (`WEI` through `ALPH`) with plain-English narrator bars, a 5-slide executive deck (`docs/slides.pdf`), and a `24" × 36"` conference poster (`docs/poster.pdf`). |
| **Reproducibility & Engineering** | Fully containerized via `Dockerfile` and `docker-compose.yml` with pinned dependencies (`requirements.txt`), an end-to-end Jupyter walkthrough (`notebook/final_analysis.ipynb`), and 87 automated unit tests. |
