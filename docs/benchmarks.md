# Benchmark Attribution Framework: Dual-Benchmark Architecture

## Overview
The quantitative evaluation suite employs a dual-benchmark attribution model comparing strategy performance against both the cap-weighted S&P 500 ETF (`SPY`) and the equal-weighted S&P 500 ETF (`RSP`).

---

## 1. Dual-Benchmark Rationale

| Dimension | Cap-Weighted (`SPY`) | Equal-Weighted (`RSP`) |
| :--- | :--- | :--- |
| **Market Index** | S&P 500 Index (Market-Cap Weighted) | S&P 500 Equal Weight Index |
| **Primary Hurdle** | Macro market equity risk premium | Structural weighting factor control |
| **Inception Date** | January 2000 (continuous full horizon) | May 1, 2003 (`2003-05-01`) |
| **Attribution Objective** | Institutional alpha ($\alpha$) and market beta ($\beta$) | Isolates executive leadership from size premium |

### Isolating Executive Selection Alpha from Structural Weighting
The Black CEO Portfolio operates on dynamic equal weighting across active constituents:

$$w_{i, t} = \frac{1}{N_t} \quad (\text{subject to } N_t \ge 3)$$

A naive comparison solely against cap-weighted `SPY` introduces factor confounding: equal-weighted portfolios systematically underweight mega-cap tech giants and overweight smaller large-cap constituents. Comparing against `RSP` controls for this weighting mechanism, allowing researchers to determine whether outperformance stems from Black executive leadership or simply equal-weight factor tilts.

---

## 2. Inception Decoupling & Timeline Filtering

### Horizon Windows
* **$\mathcal{T}_{\text{SPY}}$**: Evaluated over the full 2000–2026 backtest horizon ($6,700+$ trading sessions).
* **$\mathcal{T}_{\text{RSP}}$**: Decoupled window anchored strictly to:

$$t \ge t_{\text{RSP, inception}} = \text{May 1, 2003}$$

### Elimination of Pre-Inception Zero-Return Distortion
Naively backfilling `RSP` returns with $0.0$ before May 1, 2003 artificially suppresses annualized volatility:

$$\sigma_{\text{flawed}} = \sqrt{\frac{1}{T_{\text{full}} - 1} \sum_{t=1}^{T_{\text{full}}} (r_t - \bar{r})^2} \ll \sigma_{\text{true}}$$

The isolation engine (`isolate_rsp_benchmark_series`) explicitly truncates the strategy and benchmark to identical trading sessions where $t \ge \text{2003-05-01}$ and eliminates pre-inception zero artifacts, preserving market volatility ground truth.

---

## 3. Comparative Metric Formulations

### Beta ($\beta$)
$$\beta = \frac{\text{Cov}(r_p, r_b)}{\text{Var}(r_b)}$$

### Annualized Jensen's Alpha ($\alpha$)
$$\alpha_{\text{annual}} = 252 \cdot \left[ (\bar{r}_p - r_{f, \text{daily}}) - \beta \cdot (\bar{r}_b - r_{f, \text{daily}}) \right]$$

### Annualized Tracking Error ($\text{TE}$)
$$\text{TE} = \sqrt{252} \cdot \sqrt{\frac{1}{T-1} \sum_{t=1}^T \left( (r_{p, t} - r_{b, t}) - (\bar{r}_p - \bar{r}_b) \right)^2}$$
