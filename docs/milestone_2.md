# Milestone 2: Baseline Backtest Engine, Transaction Friction & Benchmark Decoupling

## 1. Objective & Quantitative Mandate
Milestone 2 implements the event-driven daily backtesting simulation engine (`src/backtest.py`) and institutional performance analytics suite (`src/metrics.py`). The baseline framework models an equal-weighted Black CEO equity portfolio with explicit concentration-defense rules during sparse-universe regimes ($N_t < 3$), inter-period weight drift, proportional transaction friction ($10\text{ bps}$ per unit turnover), and strict benchmark window isolation (`SPY` vs. `RSP`).

---

## 2. Core Modules & Data Artifacts

| Artifact / Module | Role in Pipeline | Key Outputs & Verification Coverage |
| :--- | :--- | :--- |
| `src/backtest.py` | Event-driven daily portfolio simulator | Tracks drift, turnover, 10 bps fee deductions, and exports `data/backtest_results.csv` |
| `src/metrics.py` | Institutional performance & risk attribution library | Computes CAGR, Sharpe, Sortino, Calmar, MDD, Beta, Jensen's Alpha, and Tracking Error |
| `data/backtest_results.csv` | Master daily return & wealth time series | 6,759 sessions (`2000-01-03` to `2026-11-16`) covering gross/net portfolio and benchmark curves |
| `data/summary_metrics.json` | Structured JSON performance scorecard | Machine-readable summary statistics for baseline strategy and benchmarks |
| `tests/test_backtest.py` | Backtest accounting & friction unit tests | Verifies budget unity ($\sum w_i + w_{\text{cash}} = 1$), $t=0$ zero fee, drift, and $r_{\text{net}} \le r_{\text{gross}}$ |
| `tests/test_metrics.py` | Mathematical metric accuracy unit tests | Validates hand-calculated synthetic series, zero-volatility guards, and full-$T$ downside deviation |
| `tests/test_benchmark_window.py` | Benchmark alignment & window decoupling tests | Confirms full-horizon `SPY` alignment and May 1, 2003 inception isolation for `RSP` |
| `tests/test_export.py` | Artifact schema & continuity tests | Validates CSV column contracts and JSON schema completeness |

---

## 3. Simulation Mechanics & Mathematical Specification

### 3.1 Sparse-Universe Concentration Defense ($N_t < 3$)
To prevent single-stock idiosyncratic shock domination during early or sparse periods when fewer than three qualifying firms are active, equity budget $B_t$ scales proportionally with active count $N_t$:

$$
B_t = \min\left(1.0,\; \frac{N_t}{3}\right), \qquad w_{i, t}^{\text{target}} = \begin{cases} \dfrac{B_t}{N_t}, & \text{if } I_{i, t} = 1 \text{ and } N_t > 0 \\ 0, & \text{otherwise} \end{cases}
$$

The residual unallocated capital $w_{\text{cash}, t} = 1 - B_t$ is held in zero-return cash so total portfolio weights satisfy $\sum_{i=1}^{N} w_{i, t} + w_{\text{cash}, t} = 1.0$ on every session.

### 3.2 Inter-Period Weight Drift & Proportional Friction Model
* **Pre-Rebalance Weight Drift:** Between consecutive trading sessions $t-1$ and $t$, asset weights drift organically with realized daily asset returns $r_{i, t}$:

$$
w_{i, t}^{\text{drift}} = \frac{w_{i, t-1}^{+} (1 + r_{i, t})}{1 + \sum_{j=1}^{N} w_{j, t-1}^{+} r_{j, t}}
$$

* **One-Way Turnover & 10 bps Cost Deduction:** At initial inception ($t = 0$), turnover is initialized to zero ($\text{TO}_0 = 0$). For all subsequent sessions $t \ge 1$, one-way portfolio turnover $\text{TO}_t$ and net daily return $r_{p, t}^{\text{net}}$ are computed as:

$$
\text{TO}_t = \frac{1}{2} \sum_{i=1}^{N} \left| w_{i, t}^{\text{target}} - w_{i, t}^{\text{drift}} \right|, \qquad r_{p, t}^{\text{net}} = r_{p, t}^{\text{gross}} - c \cdot \text{TO}_t \quad (c = 0.0010)
$$

### 3.3 Benchmark Window Decoupling (`SPY` vs. `RSP`)
* **`SPY` (Cap-Weighted S&P 500 ETF):** Evaluated across the complete `6,759`-session horizon (`2000-01-03` to `2026-11-16`).
* **`RSP` (Equal-Weighted S&P 500 ETF):** Because Invesco `RSP` incepted in late April 2003, pre-inception synthetic zero returns are stripped and all `RSP` comparative metrics (CAGR, volatility, Sharpe, Beta, Jensen's Alpha, Tracking Error) are isolated strictly on or after `2003-05-01`.

---

## 4. Empirical Baseline Scorecard (`2000-01-03` to `2026-11-16`, 6,759 Sessions)

| Performance Metric | Baseline Equal-Weight (Net) | `SPY` (Cap-Wtd) | `RSP` (Eq-Wtd, Post-May 2003) |
| :--- | :---: | :---: | :---: |
| **CAGR** | **13.36%** | 8.13% | 11.27% |
| **Annualized Volatility** | **42.09%** | 19.24% | 19.68% |
| **Sharpe Ratio ($R_f = 1.5\%$)** | **0.45** | 0.43 | 0.57 |
| **Sortino Ratio** | **0.84** | — | — |
| **Maximum Drawdown (MDD)** | **72.69%** | 55.19% | 59.92% |
| **Calmar Ratio** | **0.18** | — | — |
| **Beta vs. Benchmark** | — | 0.94 | 0.96 |
| **Jensen's Alpha (Annualized)** | — | 11.06% | 8.75% |
| **Cumulative Wealth Multiple** | **28.88x** | — | — |
| **Annualized Turnover / Total Friction** | **76.89% / 2.06%** | — | — |

---

## 5. Verification Summary
* **Accounting & Friction Invariants:** Confirmed $w_{\text{cash}} + \sum w_i = 1.0$, zero inception friction at $t = 0$, and $r_{p, t}^{\text{net}} \le r_{p, t}^{\text{gross}}$ across all 6,759 sessions.
* **Metric & Window Parity:** Verified full-$T$ downside deviation calculation and exact `2003-05-01` start-date isolation for `RSP`.
