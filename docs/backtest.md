# Backtest Engine: Specification & Institutional Friction Model

## Overview
The vectorized backtesting engine (`src/backtest.py`) simulates point-in-time portfolio returns across the dynamic Black CEO equity universe. It incorporates dynamic equal weighting, cash-drag concentration defenses for small universes, and institutional-grade transaction cost friction based on one-way turnover.

---

## 1. Portfolio Allocation & Cash-Drag Defense

### Vectorized Active Constituent Count ($N_t$)
For each trading session $t \in \mathcal{T}_{\text{SPY}}$:

$$N_t = \sum_{i=1}^{|\mathcal{U}|} M_{t, i}$$

Where $M_{t, i} \in \{0, 1\}$ is the lagged eligibility indicator from `universe_mask.csv`.

### Target Weight Allocation ($w_{i, t}$)
To prevent idiosyncratic single-stock concentration risk when universe breadth is low ($N_t < 3$), equity allocations are scaled proportionally against a 3-stock minimum diversification threshold:

$$w_{i, t} = \begin{cases} 
\frac{1}{N_t}, & \text{if } N_t \ge 3 \;\land\; M_{t, i} = 1 \\ 
\frac{1}{3}, & \text{if } 1 \le N_t < 3 \;\land\; M_{t, i} = 1 \\ 
0, & \text{otherwise} 
\end{cases}$$

### Cash Buffer & Risk-Free Yield
Unallocated portfolio weight is held in cash:

$$w_{\text{cash}, t} = 1.0 - \sum_{i=1}^{|\mathcal{U}|} w_{i, t}$$

The daily risk-free return $R_{f, \text{daily}}$ compounds from the annualized risk-free rate $R_{f, \text{annual}}$ assuming 252 sessions:

$$R_{f, \text{daily}} = (1 + R_{f, \text{annual}})^{\frac{1}{252}} - 1$$

---

## 2. Gross Return Generation

The gross portfolio daily return $R_{p, t}^{\text{gross}}$ reflects active equity holdings and the risk-free cash buffer:

$$R_{p, t}^{\text{gross}} = \sum_{i=1}^{|\mathcal{U}|} w_{i, t} \cdot r_{i, t} + w_{\text{cash}, t} \cdot R_{f, \text{daily}}$$

Where $r_{i, t} = \frac{P_{i, t}}{P_{i, t-1}} - 1$ is the daily arithmetic return from `prices_daily.csv`.

---

## 3. Institutional Trading Cost & Friction Model

### Inter-Period Weight Drift
Between rebalancing intervals, asset allocations drift away from target weights due to differential asset price movements. At the close of trading session $t$, the post-return drifted weight $w_{i, t}^{\text{drift}}$ is:

$$w_{i, t}^{\text{drift}} = w_{i, t} \cdot \frac{1 + r_{i, t}}{1 + R_{p, t}^{\text{gross}}}$$

The corresponding drifted cash position is:

$$w_{\text{cash}, t}^{\text{drift}} = 1.0 - \sum_{i=1}^{|\mathcal{U}|} w_{i, t}^{\text{drift}}$$

### One-Way Portfolio Turnover ($\text{TO}_t$)
At trading session $t$, rebalancing requires trading from the prior drifted weights $w_{i, t-1}^{\text{drift}}$ to the new target weights $w_{i, t}$. The one-way portfolio turnover $\text{TO}_t$ measures the half-sum of all absolute rebalancing adjustments across equities and cash:

$$\text{TO}_t = \frac{1}{2} \left( \sum_{i=1}^{|\mathcal{U}|} \left| w_{i, t} - w_{i, t-1}^{\text{drift}} \right| + \left| w_{\text{cash}, t} - w_{\text{cash}, t-1}^{\text{drift}} \right| \right)$$

### Inception Capitalization Adjustment ($t = 0$)
At inception ($t = 0$), deploying seed capital from cash into the opening portfolio constituents constitutes fund onboarding rather than operational strategy turnover. To prevent artificial drag on day 0, turnover is strictly anchored:

$$\text{TO}_0 = 0.0$$

### 10 bps Transaction Fee Deduction
Execution costs, bid-ask spread crossing, and exchange fees are modeled at a flat 10 basis points ($c = 10 \text{ bps} = 0.0010$) per unit of one-way turnover:

$$\text{Fee}_t = c \cdot \text{TO}_t$$

$$\text{Fee}_0 = 0.0$$

### Net Portfolio Return & Compounding Wealth
Net daily portfolio return $R_{p, t}^{\text{net}}$ reflects gross returns after deducting trading friction:

$$R_{p, t}^{\text{net}} = R_{p, t}^{\text{gross}} - \text{Fee}_t$$

Cumulative portfolio wealth $W_t$ compounds as:

$$W_t = W_0 \cdot \prod_{\tau=1}^t \left( 1 + R_{p, \tau}^{\text{net}} \right) \quad \text{with } W_0 = 1.0$$

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

