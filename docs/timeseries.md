# Parametric Time-Series Volatility & Regime Forecasting (`src/timeseries.py`)

## 1. Econometric Specifications
1. **Symmetric $\text{GARCH}(1,1)$ Maximum Likelihood (`fit_garch_11`):**
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \qquad \alpha + \beta < 1$$
   * **Shock Persistence:** $\alpha + \beta$ measures variance decay speed; half-life in trading sessions is $\text{HL} = \frac{\ln(0.5)}{\ln(\alpha + \beta)}$.
   * **Unconditional Annualized Volatility:** $\sigma_{\text{uncond}} = \sqrt{252 \cdot \frac{\omega}{1 - \alpha - \beta}}$.
2. **Asymmetric $\text{GJR-GARCH}(1,1)$ Leverage Model (`fit_gjr_garch_11`):**
   $$\sigma_t^2 = \omega + \left(\alpha + \gamma \mathbb{I}_{\epsilon_{t-1} < 0}\right)\epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
   * Tests whether negative equity shocks ($\gamma > 0$) amplify conditional variance more strongly than positive upside breakouts.
3. **Causal Walk-Forward 1-Step-Ahead Volatility-Targeted Overlay (`run_walk_forward_garch_overlay`):**
   * Refits $\text{GARCH}(1,1)$ on a strictly causal expanding window (`252`-day initial burn-in, `63`-day refit cadence) and scales long-only exposure $w_{t+1} = \min\!\left(1.0,\, \frac{\sigma_{\text{target}}}{\hat{\sigma}_{t+1|t}}\right)$ net of `10 bps` turnover friction, evaluated via **QLIKE loss**.

## 2. Exported JSON Artifact Schema (`data/timeseries_diagnostics.json`)
* `horizon_start` / `horizon_end` / `total_sessions`: Sample metadata (`2000-01-03` to `2026-11-20`, `6,763` sessions).
* `garch_1_1` & `gjr_garch_1_1`: Per-series parameter dictionaries (`mu`, `omega`, `alpha`, `gamma`, `beta`, `persistence`, `half_life_days`, `unconditional_ann_vol`, `log_likelihood`).
* `subperiod_attribution`: `pre_2010` vs. `post_2010` CAGR, Sharpe, Sortino, MDD, and GARCH persistence.
* `macro_regimes`: Six historical macroeconomic windows (`Dot-Com Bust`, `Global Financial Crisis`, `ZIRP & QE`, `COVID-19 Shock`, `2022–2023 Inflation`, `AI & High-Rate Regime`).
* `walk_forward_vol_overlay`: Out-of-sample QLIKE loss, unscaled vs. 20% vol-targeted CAGR, annualized volatility, Sharpe ratio, and MDD.
* `arima_predictability`: Ljung-Box serial correlation $p$-values and walk-forward $\text{ARIMA}(1,0,1)$ / $\text{ARIMAX}(1,0,1)$ OOS RMSE, $R^2$, and directional accuracy.
