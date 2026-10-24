# Quantitative Methodology & Backtesting Architecture

## 1. Point-in-Time Universe Construction
To eliminate survivorship and look-ahead bias, an asset $i$ is investable at date $t$ if and only if:
  $$M_{t, i} = \mathbf{1}_{[\text{start\_date}_i \le t \le \text{end\_date}_i]}$$
  
Assets outside this tenure window receive an allocation weight of zero.



## 2. Rebalancing and Weighting Schemes
- **Equal Weight ($1/N_t$):**
  $$w_{i, t} = \frac{M_{t, i}}{\sum_{j=1}^K M_{t, j}}$$
  If $N_t = \sum_j M_{t, j} = 0$, portfolio return defaults to the daily risk-free rate ($R_{f, t}$).

## 3. Transaction Costs & Friction
Turnover is tracked at each rebalance period $\tau$:
$$\text{Turnover}_\tau = \sum_{i} |w_{i, \tau} - w_{i, \tau^-}|$$
A flat transaction cost of $c = 10\text{ bps}$ ($0.0010$) is applied to turnover:
$$R_{p, \tau}^{\text{net}} = R_{p, \tau}^{\text{gross}} - c \cdot \text{Turnover}_\tau$$

## 4. Factor Attribution
Excess returns are regressed against Kenneth French's 5 factors:
$$R_{p, t} - R_{f, t} = \alpha + \beta_1(R_{m, t} - R_{f, t}) + \beta_2\text{SMB}_t + \beta_3\text{HML}_t + \beta_4\text{RMW}_t + \beta_5\text{CMA}_t + \epsilon_t$$
