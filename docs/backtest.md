# Documentation: `src/backtest.py`

## Overview
`src/backtest.py` is the execution and risk-profiling engine. It applies the dynamic boolean mask $M_{t,i}$ from `src/universe.py` across daily asset returns, applies dynamic rebalancing, models cash drag during low/zero-constituent periods, deducts transaction turnover friction, and computes standardized risk-adjusted performance metrics against market benchmarks (`SPY` and `RSP`).

---

## Mathematical Specification

### 1. Dynamic Constituent Count & Weighting ($1/N_t$)
On each trading day $t \in \{1, \dots, T\}$, the count of actively investable Black-led companies is:
$$N_t = \sum_{i=1}^K M_{t, i}$$

The target allocation weight for asset $i$ on day $t$ is:
$$w_{i, t} = \begin{cases} 
\frac{1}{N_t} & \text{if } M_{t, i} = 1 \text{ and } N_t > 0 \\ 
0 & \text{otherwise} 
\end{cases}$$

### 2. Cash Drag Handling ($N_t = 0$)
Let $R_{f, \text{ann}} = 0.03$ (3.0% annualized risk-free rate). The daily compounding risk-free return is:
$$R_{f, t} = (1 + R_{f, \text{ann}})^{1/252} - 1$$

If on date $t$, active constituent count $N_t = 0$, the strategy allocates 100% of capital to the risk-free cash buffer:
$$R_{p, t}^{\text{gross}} = \begin{cases} 
\sum_{i=1}^K w_{i, t} \cdot R_{i, t} & \text{if } N_t > 0 \\ 
R_{f, t} & \text{if } N_t = 0 
\end{cases}$$

### 3. Turnover & Transaction Cost Friction
Whenever the investable universe updates (a CEO enters or steps down), the portfolio must reallocate capital across constituents. 

Let $w_{i, t}^-$ represent the pre-rebalance weight of asset $i$ on day $t$ (drifted by price return $R_{i, t-1}$):
$$w_{i, t}^- = \frac{w_{i, t-1} (1 + R_{i, t-1})}{\sum_{j=1}^K w_{j, t-1} (1 + R_{j, t-1})}$$

For daily equal-weight rebalancing, the daily one-way portfolio turnover is:
$$\text{Turnover}_t = \frac{1}{2} \sum_{i=1}^K |w_{i, t} - w_{i, t}^-|$$

Applying a realistic institutional transaction friction of $c = 10\text{ bps} = 0.0010$ (covering bid-ask spread and exchange liquidity fees):
$$R_{p, t}^{\text{net}} = R_{p, t}^{\text{gross}} - c \cdot \text{Turnover}_t$$

### 4. Risk-Adjusted Metrics Formulation
* **CAGR (Compound Annual Growth Rate):**
  $$\text{CAGR} = \left(\prod_{t=1}^T (1 + R_{p, t}^{\text{net}})\right)^{\frac{252}{T}} - 1$$

* **Annualized Volatility ($\sigma_{\text{ann}}$):**
  $$\sigma_{\text{ann}} = \sigma(R_{p}^{\text{net}}) \cdot \sqrt{252}$$

* **Sharpe Ratio:**
  $$\text{Sharpe} = \frac{\text{Mean}(R_{p, t}^{\text{net}} - R_{f, t})}{\sigma(R_{p, t}^{\text{net}})} \cdot \sqrt{252}$$

* **Sortino Ratio (Downside Semi-Variance):**
  $$\sigma_{\text{down}} = \sqrt{\frac{1}{T}\sum_{t=1}^T \left(\min(0, R_{p, t}^{\text{net}} - R_{f, t})\right)^2} \cdot \sqrt{252}$$
  $$\text{Sortino} = \frac{\text{Mean}(R_{p, t}^{\text{net}} - R_{f, t}) \cdot \sqrt{252}}{\sigma_{\text{down}}}$$

* **Maximum Drawdown (MDD) & Calmar Ratio:**
  $$W_\tau = \prod_{t=1}^\tau (1 + R_{p, t}^{\text{net}}), \quad \text{MDD} = \min_{\tau \in [1, T]} \left(\frac{W_\tau - \max_{s \le \tau} W_s}{\max_{s \le \tau} W_s}\right)$$
  $$\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$