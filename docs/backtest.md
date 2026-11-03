# Base Backtest Engine Specification

## Overview
The base backtesting engine (`src/backtest.py`) implements a point-in-time, vectorized return simulation for the dynamic Black CEO equity universe.

---

## Mathematical Formulation

### 1. Constituent Returns
Daily simple arithmetic return for asset $i$ on trading day $t$:

$$r_{i, t} = \frac{P_{i, t}}{P_{i, t-1}} - 1$$

Where $P_{i, t}$ is the split- and dividend-adjusted closing price from `prices_daily.csv`.

### 2. Dynamic Equal-Weight Allocation ($1/N_t$)
Given the lagged investability mask $M_{t, i} \in \{0, 1\}$, the active constituent count is:

$$N_t = \sum_{i=1}^{|\mathcal{U}|} M_{t, i}$$

Constituent asset weights $w_{i, t}$ are defined by:

$$w_{i, t} = \begin{cases} \frac{1}{N_t}, & \text{if } M_{t, i} = 1 \text{ and } N_t > 0 \\ 0, & \text{otherwise} \end{cases}$$

### 3. Dynamic Cash Drag Allocation ($N_t = 0$)
When no constituent CEOs meet the active tenure criteria ($N_t = 0$), portfolio capital is held in cash:

$$w_{\text{cash}, t} = 1 - \sum_{i=1}^{|\mathcal{U}|} w_{i, t}$$

The daily risk-free rate $R_{f, \text{daily}}$ is derived from the annual risk-free yield $R_{f, \text{annual}}$ assuming 252 trading sessions:

$$R_{f, \text{daily}} = (1 + R_{f, \text{annual}})^{\frac{1}{252}} - 1$$

### 4. Total Daily Portfolio Return
$$R_{p, t} = \sum_{i=1}^{|\mathcal{U}|} w_{i, t} \cdot r_{i, t} + w_{\text{cash}, t} \cdot R_{f, \text{daily}}$$

### 5. Cumulative Compounding Wealth Series
$$W_t = W_0 \cdot \prod_{\tau=1}^t (1 + R_{p, \tau}) \quad \text{with } W_0 = 1.0$$