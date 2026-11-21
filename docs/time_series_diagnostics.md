# Time-Series Volatility, Regime Attribution & Predictability Diagnostics (2000–2026)

## 1. Executive Summary
To verify whether the Black CEO Equal-Weight (`Black_CEO_Net`) and Convex-Quality (`Convex_Quality_Net`) portfolios generate genuine structural performance rather than early-2000s small-sample artifacts or short-horizon autocorrelation, we evaluated the 6,763-session daily return series (`2000-01-03` to `2026-11-20`) across five parametric and out-of-sample econometric frameworks:
1. **Symmetric $\text{GARCH}(1,1)$ & Asymmetric $\text{GJR-GARCH}(1,1)$ Maximum Likelihood Estimation** (`src/timeseries.py`)
2. **Structural Break Sub-Period Attribution (`Pre-2010` vs. `Post-2010`)** (`src/timeseries.py`)
3. **Six Macroeconomic & Rate Event Regime Windows (`2000–2026`)** (`src/macro_regimes.py`)
4. **Causal Walk-Forward 1-Step-Ahead $\text{GARCH}(1,1)$ Volatility-Targeted Overlay** (`src/timeseries.py`)
5. **Ljung-Box Serial Correlation & Walk-Forward $\text{ARIMA}(1,0,1)$ / $\text{ARIMAX}(1,0,1)$ Predictability Tests** (`src/arima_diagnostics.py`)

---

## 2. Full-Horizon Conditional Volatility & Leverage Effects (`2000–2026`)

### 2.1 Symmetric $\text{GARCH}(1,1)$ MLE
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Series | $\alpha$ (ARCH Shock) | $\beta$ (Persistence) | $\alpha + \beta$ | Shock Half-Life | Unconditional Ann. Vol |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Black_CEO_Net`** | `0.2508` | `0.7482` | `0.9990` | `691.8d` | `201.76%` |
| **`Convex_Quality_Net`** | `0.2057` | `0.7933` | `0.9990` | `692.1d` | `130.96%` |
| **`SPY_Daily`** | `0.1277` | `0.8528` | `0.9805` | `35.2d` | `18.57%` |

### 2.2 Asymmetric $\text{GJR-GARCH}(1,1)$ Leverage MLE
$$\sigma_t^2 = \omega + \left(\alpha + \gamma \mathbb{I}_{\epsilon_{t-1} < 0}\right)\epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Series | $\alpha$ (Upside Shock) | $\gamma$ (Downside Leverage) | $\beta$ | Effective Persistence ($\alpha + \frac{1}{2}\gamma + \beta$) | Half-Life | Unconditional Ann. Vol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Black_CEO_Net`** | `0.2673` | `0.0629` | `0.7001` | `0.9989` | `623.2d` | `221.53%` |
| **`Convex_Quality_Net`** | `0.1691` | `0.2047` | `0.7275` | `0.9990` | `670.5d` | `169.03%` |
| **`SPY_Daily`** | `0.0326` | `0.1929` | `0.8500` | `0.9790` | `32.6d` | `18.48%` |

* **Key Insight (Two-Sided Volatility vs. Downside Leverage):** Unlike `SPY_Daily` ($\alpha = 0.0326, \gamma = 0.1929$), where almost all variance expansion is triggered by downside equity sell-offs, `Black_CEO_Net` exhibits near-symmetric shock response ($\alpha = 0.2673$ vs. $\gamma = 0.0629$). Upside earnings surprises and breakouts drive nearly as much conditional variance expansion as drawdowns, explaining why the full-horizon Sortino Ratio (`0.84`) is nearly double the Sharpe Ratio (`0.45`). Conversely, `Convex_Quality_Net` mirrors broad equity leverage asymmetry ($\gamma = 0.2047 > \alpha = 0.1691$).

---

## 3. Structural Break Attribution: Pre-2010 vs. Post-2010 Regimes

Splitting the backtest at **January 1, 2010** isolates the sparse early-2000s universe ($N_t \in [1, 5]$) from the broader modern era ($N_t$ expanding up to $17$):

| Regime | Series | Sessions ($n$) | CAGR | Sharpe | Sortino | Max Drawdown | GARCH $\alpha + \beta$ | Shock Half-Life |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-2010** (`2000–2009`) | `Black_CEO_Net` | `2,514` | `+7.33%` | `0.33` | `0.51` | `72.69%` | `0.9936` | `107.6d` |
| **Pre-2010** (`2000–2009`) | `Convex_Quality_Net` | `2,514` | `+1.77%` | `0.17` | `0.27` | `71.87%` | `0.9990` | `691.5d` |
| **Pre-2010** (`2000–2009`) | `SPY_Daily` | `2,515` | `-0.91%` | `0.00` | `0.01` | `55.19%` | `0.9925` | `92.7d` |
| **Post-2010** (`2010–2026`) | `Black_CEO_Net` | `4,241` | `+17.07%` | `0.50` | `1.03` | `53.21%` | `0.9395` | `11.1d` |
| **Post-2010** (`2010–2026`) | `Convex_Quality_Net` | `4,241` | `+14.38%` | `0.55` | `0.93` | `44.07%` | `0.9982` | `390.6d` |
| **Post-2010** (`2010–2026`) | `SPY_Daily` | `4,248` | `+13.39%` | `0.73` | `1.03` | `33.72%` | `0.9668` | `20.5d` |

* **Resolution of Full-Horizon Unit-Root Persistence:** The full-horizon `0.9990` boundary persistence on `Black_CEO_Net` is an artifact of fitting a single variance regime across a structural break. Post-2010, as constituent breadth expanded, `Black_CEO_Net` persistence dropped to **`0.9395`** (below SPY's `0.9668`), compressing shock half-life by **90%** (`107.6d` $\rightarrow$ `11.1d`).
* **Post-2010 Return Acceleration:** Rather than fading as institutional coverage grew, `Black_CEO_Net` accelerated to a **`17.07%` CAGR** post-2010 (`+368 bps/yr` over SPY) with a `1.03` Sortino Ratio, while `Convex_Quality_Net` achieved the strongest strategy Sharpe (`0.55`) and lowest drawdown (`44.07%`).

---

## 4. Macroeconomic & Rate Regime Attribution (`2000–2026`)

| Macro Event Window | Date Range | Active Breadth ($N_{\min}$–$N_{\max}$) | `Black_CEO_Net` Cum. Return | `SPY_Daily` Cum. Return | `Black_CEO_Net` Max DD | Trough-to-Peak Recovery | Downside Capture vs. SPY |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dot-Com Bust & 2001 Recession** | `2000-03-10` – `2002-10-09` | `4–5` | **`+12.36%`** | `-42.64%` | `33.37%` | `59d` | `0.80x` |
| **Global Financial Crisis** | `2007-10-09` – `2009-03-09` | `5–5` | **`-67.75%`** | `-54.77%` | `68.17%` | `Unrec. in window` | `0.99x` |
| **ZIRP & Quantitative Easing Era** | `2009-03-10` – `2020-02-19` | `4–10` | **`+1,002.23%`** | `+520.77%` | `53.21%` | `491d` | `0.90x` |
| **COVID-19 Shock & Stimulus Surge** | `2020-02-20` – `2021-12-31` | `9–12` | **`+171.42%`** | `+44.91%` | `46.98%` | `61d` | `1.28x` |
| **2022–2023 Inflation & Rate-Hike Cycle** | `2022-01-03` – `2023-10-27` | `11–16` | **`-19.95%`** | `-11.16%` | `26.89%` | `Unrec. in window` | `1.22x` |
| **AI & High-Rate Regime** | `2023-10-30` – `2026-11-20` | `0–17` | **`+110.37%`** | `+74.17%` | `17.65%` | `95d` | `0.79x` |

---

## 5. Walk-Forward 1-Step-Ahead $\text{GARCH}(1,1)$ Volatility-Targeted Overlay

Using strictly causal expanding-window $\text{GARCH}(1,1)$ refits (252-day initial burn-in, 63-day refit cadence, 10 bps turnover friction) to scale long-only equity exposure $w_{t+1} = \min\!\left(1.0,\, \frac{20\%}{\hat{\sigma}_{t+1|t}}\right)$ across `6,511` out-of-sample sessions (`QLIKE Loss = 1.6707`):

| Portfolio Variant (OOS Horizon) | CAGR | Realized Ann. Vol | Sharpe Ratio ($R_f=1.5\%$) | Maximum Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **Unscaled `Black_CEO_Net` (100% Invested)** | `14.49%` | `42.20%` | `0.47` | `72.69%` |
| **GARCH Vol-Targeted Overlay (Target = `20.00%`)** | `9.52%` | **`19.95%`** | **`0.48`** | **`51.22%`** |

* Scaling exposure inversely to 1-step-ahead conditional volatility hits the `20.00%` annualized risk budget within **5 bps** (`19.95%` realized), compresses Maximum Drawdown by **2,147 bps** (`72.69%` $\rightarrow$ `51.22%`, outperforming SPY's `55.19%` MDD), and slightly improves net Sharpe (`0.47` $\rightarrow$ `0.48`) after transaction costs.

---

## 6. Walk-Forward $\text{ARIMA}(1,0,1)$ / $\text{ARIMAX}(1,0,1)$ Return Predictability

Evaluated across `6,511` out-of-sample sessions on daily net excess returns ($r_{p,t} - r_{f,t}$), with `ARIMAX` conditioned on lagged market excess return ($Mkt\text{-}RF_{t-1}$) and lagged active constituent breadth ($N_{t-1}$):

* **Ljung-Box Serial Correlation $p$-values:** Lag 1 = `0.1164` (no significant next-day linear autocorrelation); Lags 5, 10, 21 = `< 1e-20` (reflecting multi-day volatility clustering captured by GARCH).
* **Expanding-Mean Benchmark OOS RMSE:** `0.02659`
* **Walk-Forward $\text{ARIMA}(1,0,1)$:** OOS RMSE = `0.02660` | $\text{OOS } R^2 = -0.0550\%$ | Directional Accuracy = `51.21%`
* **Walk-Forward $\text{ARIMAX}(1,0,1)$:** OOS RMSE = `0.02662` | $\text{OOS } R^2 = -0.2598\%$ | Directional Accuracy = `51.68%`

**Econometric Conclusion:** Daily Black CEO portfolio excess returns behave as an informationally efficient martingale at 1-day forecast horizons ($\text{OOS } R^2 \approx 0\%$). Outperformance is driven by structural factor/regime attribution and volatility-managed sizing rather than short-horizon linear return autocorrelation.
