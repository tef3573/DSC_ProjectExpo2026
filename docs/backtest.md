# Backtest Engine: Cash-Drag Concentration Defense

## Overview
When active constituent count $N_t$ is low, standard equal-weighting ($1/N_t$) produces single-stock concentration risk. The engine enforces a minimum diversification threshold ($K = 3$) by scaling equity exposure and allocating unassigned capital to cash.

---

## Mathematical Formulation

### 1. Vectorized Active Count ($N_t$)
For each session $t$:

$$N_t = \sum_{i=1}^{|\mathcal{U}|} M_{t, i}$$

### 2. Proportional Equity Scaling ($N_t < 3$)
Constituent weight $w_{i, t}$ for eligible asset $i$ is governed by:

$$w_{i, t} = \begin{cases} 
\frac{1}{N_t}, & \text{if } N_t \ge 3 \;\land\; M_{t, i} = 1 \\ 
\frac{1}{3}, & \text{if } 1 \le N_t < 3 \;\land\; M_{t, i} = 1 \\ 
0, & \text{otherwise} 
\end{cases}$$

### 3. Cash Buffer Allocation
Total equity exposure and residual cash buffer satisfy:

$$W_{\text{equity}, t} = \sum_{i=1}^{|\mathcal{U}|} w_{i, t} = \begin{cases} 1.0, & N_t \ge 3 \\ \frac{N_t}{3}, & 1 \le N_t < 3 \\ 0.0, & N_t = 0 \end{cases}$$

$$w_{\text{cash}, t} = 1.0 - W_{\text{equity}, t}$$

### 4. Risk-Free Yield Routing
Unallocated cash earns the daily risk-free rate derived from annual yield $R_{f, \text{annual}}$:

$$R_{f, \text{daily}} = (1 + R_{f, \text{annual}})^{\frac{1}{252}} - 1$$

$$R_{p, t} = \sum_{i=1}^{|\mathcal{U}|} w_{i, t} \cdot r_{i, t} + w_{\text{cash}, t} \cdot R_{f, \text{daily}}$$
