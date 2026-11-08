# Vectorized Backtest Engine: Architectural & Quantitative Specification

## Overview
The vectorized backtest engine (`src/backtest.py`) implements an institutional simulation framework evaluating the performance of publicly traded equities led by Black CEOs across the 2000–2026 horizon. The engine eliminates survivorship bias and look-ahead bias, incorporates realistic execution lag and capital concentration defenses, models daily weight drift, deducts transaction friction, and generates dual-benchmark attribution against `SPY` and `RSP`.

---

## 1. Portfolio Construction Mechanics

### Dynamic Equal-Weight Allocation ($w_{i, t}$)
For active constituents defined by the point-in-time universe mask $M_{t, i} \in \{0, 1\}$, target weights are dynamically scaled:

$$N_t = \sum_{i=1}^{|\mathcal{U}|} M_{t, i}$$

### Diversification & Cash-Drag Defense ($N_t < 3$)
To avoid extreme single-stock concentration risk when the active constituent count falls below the diversification threshold ($N_{\text{min}} = 3$), capital is scaled proportionally and unallocated wealth is held in cash yielding the risk-free rate $r_{f, \text{daily}}$:

$$w_{i, t} = \begin{cases} 
\frac{1}{N_t} & \text{if } N_t \ge 3 \quad (\text{Cash}_t = 0.0) \\
\frac{1}{3} & \text{if } 0 < N_t < 3 \quad \left(\text{Cash}_t = 1.0 - \frac{N_t}{3}\right) \\
0.0 & \text{if } N_t = 0 \quad (\text{Cash}_t = 1.0)
\end{cases}$$

This satisfies the unity capital constraint at all sessions:

$$\sum_{i=1}^{|\mathcal{U}|} w_{i, t} + \text{Cash}_t = 1.0 \quad \forall \; t$$

---

## 2. Weight Drift & Execution Friction

### Inter-Period Weight Drift
Between daily rebalance events, price movements naturally drift constituent portfolio allocations. The drifted weight of asset $i$ at the close of trading session $t$ prior to session $t+1$ rebalancing is:

$$w_{i, t}^{\text{drifted}} = \frac{w_{i, t} \cdot (1 + r_{i, t})}{1 + r_{p, t}^{\text{gross}}}$$

Where the gross daily strategy return $r_{p, t}^{\text{gross}}$ includes the cash yield:

$$r_{p, t}^{\text{gross}} = \sum_{i=1}^{|\mathcal{U}|} w_{i, t} r_{i, t} + \text{Cash}_t \cdot r_{f, \text{daily}}$$

### One-Way Turnover & Transaction Cost Deduction
Rebalancing adjusts drifted weights $w_{i, t-1}^{\text{drifted}}$ back to target weights $w_{i, t}$. One-way turnover is defined as:

$$\tau_t = \frac{1}{2} \left( \sum_{i=1}^{|\mathcal{U}|} \left| w_{i, t} - w_{i, t-1}^{\text{drifted}} \right| + \left| \text{Cash}_t - \text{Cash}_{t-1}^{\text{drifted}} \right| \right)$$

* **Inception Invariant:** On the initial backtest session ($t = 0$), position inception incurs $\tau_0 = 0.0$ and $C_0 = 0.0$.
* **Institutional Friction:** A fee deduction of 10 basis points ($f = 0.0010$) is applied to one-way turnover:

$$C_t = \tau_t \cdot f$$

$$r_{p, t}^{\text{net}} = r_{p, t}^{\text{gross}} - C_t \le r_{p, t}^{\text{gross}}$$

---

## 3. Dual-Benchmark Attribution Model

The strategy is evaluated concurrently across two decoupled benchmarks:

1. **Cap-Weighted Benchmark (`SPY`)**:
   * Evaluated across the full 2000–2026 backtest horizon ($6,700+$ trading sessions).
   * Measures broad equity market beta ($\beta$) and institutional Jensen's Alpha ($\alpha$).
2. **Equal-Weighted Benchmark (`RSP`)**:
   * Decoupled evaluation window starting at `RSP` inception on **May 1, 2003** ($t \ge \text{2003-05-01}$).
   * Pre-inception synthetic zero returns are filtered out to preserve market ground truth volatility ($16.0\% \le \sigma_{\text{ann}} \le 22.0\%$).
   * Controls for the equal-weight size factor premium, isolating executive leadership selection alpha from weighting mechanics.

---

## 4. Export Artifacts & Data Contracts

The backtest pipeline exports two standard artifacts for visualization modules, dashboards, and research reporting:

### 1. Daily Time Series (`data/backtest_results.csv`)
A CSV indexed by NYSE trading calendar sessions containing:
* `Black_CEO_Gross`: Daily arithmetic return before transaction fees.
* `Black_CEO_Net`: Daily return net of 10 bps turnover friction.
* `Black_CEO_Cumulative`: Normalized compounding wealth curve ($W_0 = 1.0$).
* `Cash_Weight`: Daily fraction of defensive cash allocation.
* `Active_Constituents_Nt`: Point-in-time active constituent count ($N_t$).
* `One_Way_Turnover`: Daily portfolio rebalancing turnover ($\tau_t$).
* `Transaction_Costs`: Daily deducted transaction fee ($C_t$).
* `Strategy_Drawdown`: Daily percentage drawdown from running high-water peak ($DD_t \le 0$).
* `SPY_Daily`, `SPY_Cumulative`, `SPY_Drawdown`: Cap-weighted market benchmark curves.
* `RSP_Daily`, `RSP_Cumulative`, `RSP_Drawdown`: Equal-weighted market benchmark curves.

### 2. Summary Metrics JSON (`data/summary_metrics.json`)
A structured dictionary storing:
* `metadata`: Backtest start/end dates, total sessions, $R_f$ rate, fee schedule.
* `strategy_performance`: Net/gross CAGR, annualized volatility, Sharpe ratio, Sortino ratio ($T$-sample), Calmar ratio, MDD, cumulative wealth multiple, annualized turnover, and total friction cost.
* `benchmark_spy_full_horizon`: Full 2000–2026 comparative metrics ($\alpha$, $\beta$, tracking error, relative CAGR/vol/MDD).
* `benchmark_rsp_isolated`: Post-May 2003 isolated equal-weight comparative metrics.
