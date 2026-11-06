# Portfolio Performance & Risk Metrics: Analytical Specification

## Overview
The performance analytics suite (`src/metrics.py`) provides risk-adjusted return evaluation, downside semi-deviation tracking, drawdown analytics, and benchmark attribution adhering to institutional quantitative standards.

---

## 1. Annualized Sharpe Ratio

The annualized Sharpe ratio evaluates mean portfolio excess returns relative to total sample return volatility. 

### Formulation
Returns scale linearly with time ($252$ trading days/year) while sample volatility scales with the square root of time ($\sqrt{252}$):

$$\text{Sharpe} = \frac{\mu_{\text{excess, annual}}}{\sigma_{\text{annual}}} = \frac{\mu_{\text{excess, daily}} \cdot 252}{\sigma_{\text{daily}} \cdot \sqrt{252}} = \frac{\mu_{\text{excess, daily}}}{\sigma_{\text{daily}}} \cdot \sqrt{252}$$

Where:
* $r_t$: Daily net strategy arithmetic return.
* $r_{f, \text{daily}} = (1 + r_{f, \text{annual}})^{\frac{1}{252}} - 1$: Daily compounded risk-free rate.
* $\mu_{\text{excess, daily}} = \frac{1}{T} \sum_{t=1}^T (r_t - r_{f, \text{daily}})$.
* $\sigma_{\text{daily}} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (r_t - \bar{r})^2}$.

---

## 2. Annualized Sortino Ratio ($T$-Sample Formulation)

The Sortino ratio isolates downside volatility by penalizing exclusively negative deviations from the risk-free hurdle.

### Downside Semi-Deviation ($\delta_{\text{downside}}$)
To prevent artificial inflation from counting only down days, downside semi-deviation is calculated over the full sample length $T$:

$$\delta_{\text{downside}} = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(r_t - r_{f, \text{daily}}, 0)^2}$$

### Annualized Sortino Ratio
$$\text{Sortino} = \frac{\mu_{\text{excess, daily}} \cdot 252}{\delta_{\text{downside}} \cdot \sqrt{252}} = \frac{\mu_{\text{excess, daily}}}{\delta_{\text{downside}}} \cdot \sqrt{252}$$

---

## 3. Drawdown Dynamics & Maximum Drawdown (MDD)

### Cumulative Wealth Curve ($W_t$)
With normalized starting capital $W_0 = 1.0$:

$$W_t = \prod_{\tau=1}^t (1 + r_\tau)$$

### Running High-Water Mark ($P_t$) & Drawdown ($DD_t$)
$$P_t = \max_{0 \le \tau \le t} W_\tau$$

$$DD_t = \frac{W_t - P_t}{P_t} \le 0$$

### Maximum Drawdown (MDD)
The peak-to-trough maximum observed decline across the full horizon:

$$\text{MDD} = \left| \min_{1 \le t \le T} DD_t \right|$$

---

## 4. Compound Annual Growth Rate (CAGR) & Calmar Ratio

### Compound Annual Growth Rate (CAGR)
$$\text{CAGR} = W_T^{\frac{252}{T}} - 1$$

### Calmar Ratio
The Calmar ratio measures annualized compound capital growth per unit of tail drawdown risk:

$$\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}}$$