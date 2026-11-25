# Milestone 4: Fama-French 5-Factor Attribution, Econometric Time-Series & Statistical Robustness

## 1. Objective & Quantitative Mandate
Milestone 4 evaluates whether the cumulative outperformance of the Black CEO equity strategy ($12.83\%$ net CAGR, $25.55\times$ cumulative wealth multiple, and $10.00\%$ annualized Jensen alpha vs. `SPY` across `2000-01-03` to `2026-11-24`, `6,765` sessions) represents genuine idiosyncratic skill rather than passive exposure to market, size, value, profitability, or investment style tilts. To enforce statistical rigor under fat-tailed, volatility-clustered daily equity returns, Milestone 4 combines Fama-French 5-Factor (`FF5`) OLS regression with Newey-West heteroskedasticity- and autocorrelation-consistent (`HAC`) standard errors, maximum-likelihood `GARCH(1,1)` and asymmetric `GJR-GARCH(1,1)` volatility modeling, walk-forward `ARIMA` / `ARIMAX` predictability diagnostics, macro-regime attribution, and stationary block bootstrap inference.

---

## 2. Core Modules & Data Artifacts

| Artifact / Module | Role in Pipeline | Key Outputs & Verification Coverage |
| :--- | :--- | :--- |
| `src/attribution.py` | Fama-French 5-Factor OLS regression & Newey-West HAC alpha inference engine | Ingests Kenneth French daily `FF5` factors (`6,726` decimal sessions), estimates factor loadings, formats Markdown $p$-value tables, and exports `data/factor_loadings.csv` |
| `src/bootstrap.py` | Non-parametric stationary block bootstrap & Monte Carlo inference engine | Resamples dependent return blocks to compute empirical Sharpe, CAGR, and alpha distributions and exports `data/mc_results.json` |
| `src/timeseries.py` | Conditional heteroskedasticity (`GARCH(1,1)` / `GJR-GARCH(1,1)`) & sub-period MLE engine | Estimates volatility persistence ($\alpha + \beta$), shock half-life, leverage asymmetry ($\gamma$), and Pre-2010 vs. Post-2010 stability, exporting `data/timeseries_diagnostics.json` |
| `src/arima_diagnostics.py` | Out-of-sample walk-forward return predictability & serial correlation diagnostics | Executes Ljung-Box $Q$-tests across Lags 1, 5, 10, and 21 alongside rolling `ARIMA(1,0,1)` and `ARIMAX(1,0,1)` benchmarks across `6,513` OOS sessions |
| `src/macro_regimes.py` | Macro-regime & crisis-window attribution engine (`2000–2026`) | Decomposes cumulative return, maximum drawdown, recovery duration, and downside capture across six distinct macroeconomic regimes |
| `data/factor_loadings.csv` | Structured `FF5` regression loadings, $t$-statistics, and HAC $p$-values | Machine-readable factor attribution table across weighting schemes |
| `data/mc_results.json` | Bootstrap confidence intervals and empirical $p$-value summary | Non-parametric sampling distributions for risk-adjusted performance metrics |
| `data/timeseries_diagnostics.json` | Unified econometric diagnostics payload | Consumed by Pages `4` (`ECST MACRO`), `5` (`VOL GARCH`), and `6` (`ALPH FF5`) of the interactive terminal |
| `tests/test_integration_milestone4.py` | End-to-end Milestone 4 pipeline integration suite | Verifies continuity from `data/backtest_results.csv` through `FF5` OLS regression, HAC $p$-value formatting, `factor_loadings.csv`, and `mc_results.json` |

---

## 3. Econometric & Statistical Formulation

### 3.1 Fama-French 5-Factor Regression with Newey-West HAC Inference
Daily strategy excess returns $R_{p, t} - R_{f, t}$ are regressed on the five Kenneth French daily risk factors—market excess return ($\text{Mkt-RF}$), size ($\text{SMB}$), value ($\text{HML}$), operating profitability ($\text{RMW}$), and investment conservatism ($\text{CMA}$):

$$
R_{p, t} - R_{f, t} = \alpha_d + \beta_{\text{MKT}}(\text{Mkt-RF}_t) + \beta_{\text{SMB}}\text{SMB}_t + \beta_{\text{HML}}\text{HML}_t + \beta_{\text{RMW}}\text{RMW}_t + \beta_{\text{CMA}}\text{CMA}_t + \varepsilon_t
$$

- **HAC Covariance Correction:** Because daily equity residuals exhibit conditional heteroskedasticity and serial dependence, standard errors are computed using the Newey-West HAC covariance estimator with automatic/specified lag truncation, and daily intercept $\alpha_d$ is annualized via $\alpha_{\text{ann}} = (1 + \alpha_d)^{252} - 1$ (and linear $252\,\alpha_d$ scaling for hypothesis testing).

### 3.2 Stationary Block Bootstrap Resampling (`SyntheticBootstrapEngine`)
To avoid Gaussian normality assumptions when evaluating Sharpe ratios, cumulative returns, and factor alpha, `src/bootstrap.py` implements Politis-Romano stationary block bootstrap resampling with geometric block lengths (preserving short-run autocorrelation and volatility clustering) to construct non-parametric $95\%$ confidence intervals and empirical $p$-values.

### 3.3 Conditional Volatility (`GARCH(1,1)` & `GJR-GARCH(1,1)`) & Walk-Forward `ARIMA`
- **Symmetric `GARCH(1,1)` & Asymmetric `GJR-GARCH(1,1)` MLE:** Conditional variance $\sigma_t^2$ is estimated via maximum likelihood:

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma \varepsilon_{t-1}^2 \mathbb{I}(\varepsilon_{t-1} < 0) + \beta \sigma_{t-1}^2
$$

  Full-horizon `GARCH(1,1)` persistence reaches $\alpha + \beta = 0.9983$ ($\text{half-life} = 401.4\text{d}$) for `Black_CEO_Net` vs. $0.9809$ ($\text{half-life} = 36.0\text{d}$) for `SPY`, while Post-2010 regime persistence moderates to $\alpha + \beta = 0.9061$ ($\text{half-life} = 7.0\text{d}$) as cross-sectional breadth expands ($N_t$ up to $17$).
- **Serial Correlation & Walk-Forward Predictability (`6,513` OOS Sessions):** Ljung-Box tests confirm first-order return unpredictability at Lag 1 ($p = 0.1741$, consistent with weak-form market efficiency; `ARIMA(1,0,1)` OOS $R^2 = -0.0774\%$, directional accuracy $50.84\%$; `ARIMAX(1,0,1)` OOS $R^2 = -0.0236\%$, directional accuracy $51.71\%$) alongside highly significant second-moment volatility clustering across Lags 5–21 ($p < 10^{-31}$), validating `GARCH`-family variance modeling.

---

## 4. Macro-Regime Attribution Summary (`2000–2026`)

| Macro Regime | Active Count ($N_t$) | Equal-Weight Net Cum. Return | `SPY` Cum. Return | Strategy MDD (Recovery) | Downside Capture |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dot-Com Bust & 2001 Recession | $6\text{–}7$ | $+8.16\%$ | $-42.64\%$ | $30.26\%$ ($99\text{d}$) | $0.62\times$ |
| Global Financial Crisis | $7\text{–}7$ | $-65.77\%$ | $-54.77\%$ | $66.05\%$ (Unrec.) | $0.94\times$ |
| ZIRP & Quantitative Easing Era | $6\text{–}12$ | $+960.47\%$ | $+520.77\%$ | $45.11\%$ ($420\text{d}$) | $0.82\times$ |
| COVID-19 Shock & Stimulus Surge | $10\text{–}13$ | $+141.48\%$ | $+44.91\%$ | $46.11\%$ ($61\text{d}$) | $1.26\times$ |
| 2022–2023 Inflation & Rate-Hike Cycle | $12\text{–}16$ | $-20.67\%$ | $-11.16\%$ | $26.35\%$ (Unrec.) | $1.23\times$ |
| AI & High-Rate Regime | $15\text{–}17$ | $+92.86\%$ | $+74.17\%$ | $17.65\%$ ($95\text{d}$) | $0.82\times$ |

---

## 5. Verification Summary
- **End-to-End Integration (`tests/test_integration_milestone4.py`):** Verified `FF5` OLS factor estimation, Newey-West HAC annualized alpha inference, Markdown $p$-value table generation, `factor_loadings.csv` export, and `mc_results.json` bootstrap serialization.
- **Full Regression Suite:** All `87` unit and integration tests across Milestones 1–4 pass (`OK`).
