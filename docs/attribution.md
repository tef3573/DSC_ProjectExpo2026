# Fama-French 5-Factor Attribution & Statistical Significance Framework

## 1. Econometric Model Specification
To decompose whether the Black CEO equity portfolio generates genuine idiosyncratic alpha beyond standard equity risk premia, `src/attribution.py` estimates the Fama-French 5-Factor (FF5) time-series regression across all overlapping daily sessions ($T = 6,726$ from January 2000 through September 2026):

$$
R_{p,t} - R_{f,t} = \alpha_{\text{daily}} + \beta_{\text{MKT}}(R_{m,t} - R_{f,t}) + \beta_{\text{SMB}}\text{SMB}_t + \beta_{\text{HML}}\text{HML}_t + \beta_{\text{RMW}}\text{RMW}_t + \beta_{\text{CMA}}\text{CMA}_t + \varepsilon_t
$$

where:
* $R_{p,t} - R_{f,t}$ is the daily strategy excess return over the 1-month U.S. Treasury bill rate (`RF`), aligned strictly on the Kenneth French daily trading calendar (`data/ff5_factors_daily.csv`).
* **Market (`Mkt-RF`):** Excess return on the value-weighted U.S. equity market.
* **Size (`SMB`):** Small-Minus-Big market capitalization factor.
* **Value (`HML`):** High-Minus-Low book-to-market value factor.
* **Profitability (`RMW`):** Robust-Minus-Weak operating profitability factor.
* **Investment (`CMA`):** Conservative-Minus-Aggressive asset growth factor.

---

## 2. Newey-West HAC Covariance & Hypothesis Inference
Because concentrated equity portfolios exhibit volatility clustering and mild serial correlation in daily residuals, standard OLS covariance estimates can overstate $t$-statistics. `compute_newey_west_hac_covariance()` implements the heteroskedasticity- and autocorrelation-consistent (HAC) estimator with Bartlett kernel weights $w_l = 1 - \frac{l}{L + 1}$ at lag truncation $L = 5$:

$$
\widehat{\Sigma}_{\text{HAC}} = (X^\top X)^{-1} \left( \sum_{t=1}^{T} \hat{\varepsilon}_t^2 x_t x_t^\top + \sum_{l=1}^{L} \left(1 - \frac{l}{L+1}\right) \sum_{t=l+1}^{T} \hat{\varepsilon}_t \hat{\varepsilon}_{t-l} \left(x_t x_{t-l}^\top + x_{t-l} x_t^\top\right) \right) (X^\top X)^{-1}
$$

* **Annualized Linear Alpha:** $\alpha_{\text{ann}} = 252 \cdot \hat{\alpha}_{\text{daily}}$.
* **Hypothesis Structure:** Evaluates both two-sided ($H_0: \alpha = 0$) and directional one-sided ($H_1: \alpha > 0$) significance using Student-$t$ critical values with $T - 6$ degrees of freedom.

---

## 3. Empirical Factor Loadings (`data/factor_loadings.csv`)

| Parameter | Equal-Weight (Net) Estimate | Equal-Weight HAC $t$-stat | Equal-Weight $p$-value | Convex-Quality (Net) Estimate | Convex-Quality HAC $t$-stat | Convex-Quality $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\alpha_{\text{daily}}$** | `+0.0417%` | `+1.51` | `0.1323` (`0.0661` 1-sided) | `+0.0130%` | `+0.66` | `0.5105` (`0.2553` 1-sided) |
| **$\alpha_{\text{annualized}}$** | **`+10.52%`** | **`+1.51`** | **`0.1323`** (`0.0661` 1-sided) | **`+3.28%`** | **`+0.66`** | **`0.5105`** (`0.2553` 1-sided) |
| **`Mkt-RF` ($\beta_{\text{MKT}}$)** | `+0.8888` | `+23.08` | `< 0.0001` | `+0.9299` | `+48.56` | `< 0.0001` |
| **`SMB` ($\beta_{\text{SMB}}$)** | `+0.4147` | `+5.91` | `< 0.0001` | `+0.1475` | `+3.15` | `0.0016` |
| **`HML` ($\beta_{\text{HML}}$)** | `+0.2852` | `+5.57` | `< 0.0001` | `+0.2137` | `+4.49` | `< 0.0001` |
| **`RMW` ($\beta_{\text{RMW}}$)** | `-0.1749` | `-2.24` | `0.0251` | `-0.1013` | `-2.04` | `0.0412` |
| **`CMA` ($\beta_{\text{CMA}}$)** | `-0.1112` | `-1.78` | `0.0757` | `+0.0820` | `+1.38` | `0.1683` |
| **$R^2$ / Adjusted $R^2$** | `0.2189` / `0.2183` | — | — | `0.3788` / `0.3783` | — | — |

### Key Econometric Takeaways
1. **Substantial Positive Alpha After Size/Value Controls:** Equal-Weight net returns exhibit highly significant positive loadings on small-cap (`SMB = +0.415`, $t = 5.91$) and value (`HML = +0.285`, $t = 5.57$) factors. Even after accounting for those tilts, annualized FF5 alpha remains **+10.52%** ($t_{\text{HAC}} = +1.51$, one-sided $p = 0.0661$), significant at the $10\%$ directional level.
2. **Idiosyncratic Variance & $R^2$:** Roughly $78.1\%$ of Equal-Weight daily variance is idiosyncratic ($R^2 = 0.2189$), widening HAC confidence intervals around alpha. Under `Convex-Quality`, variance compression increases factor $R^2$ to $0.3788$ and lowers `SMB` exposure from `0.415` to `0.147`.

---

## 4. Non-Parametric Monte Carlo Permutation Synthesis (`data/mc_results.json`)
To complement parametric HAC regressions, `SyntheticBootstrapEngine` (`docs/monte_carlo.md`) simulates $B = 10,000$ breadth-matched random portfolios ($N_t$ active holdings and cash buffer $B_t = \min(1.0, N_t / 3.0)$, `seed = 42`):

* **Equal-Weight CAGR ($13.36\%$ vs. $7.79\%$ Null Mean):** Ranks at the **99.77th percentile** with a right-tailed empirical $p$-value of **$0.0024$** (`95%` null bounds `[4.06%, 11.57%]`), demonstrating statistically significant compound outperformance ($p < 0.01$) against random stock selection of identical portfolio concentration.
* **Equal-Weight Sharpe ($0.4452$ vs. $0.3877$ Null Mean):** Ranks at the **75.46th percentile** ($p_{\text{right}} = 0.2455$, `95%` null bounds `[0.2232, 0.5498]`).

---

## 5. Reproducible Artifact Pipeline
Running `src/attribution.py` regenerates both persisted Milestone 4 reporting artifacts deterministically:
1. `data/factor_loadings.csv` — Tabular FF5 OLS estimates, HAC $t$-statistics, two-sided/one-sided $p$-values, and $R^2$ diagnostics (`export_factor_loadings_csv()`).
2. `data/mc_results.json` — Full $10,000$-trial percentile ladder (`p1_0` through `p99_0`), two-tailed $95\%$ bounds, percentile ranks, and Davison-Hinkley empirical $p$-values (`export_mc_results_json()`).
