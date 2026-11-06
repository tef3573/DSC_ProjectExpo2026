# Point-in-Time Dynamic Universe Mask & Execution Lag Engine

## Overview
The universe masking engine (`src/universe.py`) translates the audited executive tenure roster (`data/black_ceo_roster.csv`) into a daily boolean investability matrix $M_{t, i}$. This ensures zero survivorship bias and eliminates look-ahead bias across the 2000–2026 backtest horizon.

---

## 1. Point-in-Time Raw Indicator Matrix

For each trading session $t \in \mathcal{T}_{\text{SPY}}$ and constituent asset $i \in \mathcal{U}$:

$$I_{t, i} = \mathbf{1}_{\{t \ge \text{start\_date}_i \;\land\; t \le \text{end\_date}_i \;\land\; P_{t, i} \neq \text{NaN}\}}$$

Where:
* $\text{start\_date}_i$: Effective appointment date of the Black CEO.
* $\text{end\_date}_i$: Official departure / resignation date.
* $P_{t, i}$: Daily adjusted closing price from `prices_daily.csv`.
* $\mathcal{T}_{\text{SPY}}$: Official NYSE trading calendar anchored by the S&P 500 benchmark (`SPY`).

---

## 2. Execution Lag ($t+1$) Look-Ahead Defense

To prevent look-ahead bias and simulate realistic institutional trading, eligibility signals are lagged by 1 trading session:

$$M_{t, i} = I_{t-1, i}$$

If a CEO assumes office on date $t$, the portfolio cannot hold the asset until trading session $t+1$. Similarly, upon departure on date $t$, the asset remains held until liquidation at the close of $t+1$.

---

## 3. Active Constituent Count ($N_t$) & Diversification Threshold

The total active constituent count at time $t$ is given by:

$$N_t = \sum_{i=1}^{|\mathcal{U}|} M_{t, i}$$

* **Upper Bound Constraint:** $0 \le N_t \le |\mathcal{U}_{\text{roster}}| = 21$
* **Diversification Defense:** When $N_t < 3$, portfolio construction rules allocate unassigned capital to cash or the equal-weight benchmark (`RSP`) to prevent single-stock concentration risk.
