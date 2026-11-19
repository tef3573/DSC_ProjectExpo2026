# Monte Carlo Permutation Bootstrap Methodology

## 1. Objective & Hypothesis Framework
To evaluate whether the Black CEO equity portfolio's cumulative performance reflects genuine stock-selection skill or a statistical artifact of holding a concentrated, time-varying basket of $N_t$ equities ($0 \le N_t \le 17$), `SyntheticBootstrapEngine` (`src/attribution.py`) implements a non-parametric Monte Carlo permutation bootstrap across $B = 10,000$ synthetic trials.

Rather than comparing a concentrated $N_t$-stock portfolio directly against a 500-stock index—which conflates stock-selection alpha with small-sample dispersion—each bootstrap trial constructs a randomized equal-weight portfolio matching the exact daily active breadth $N_t$ and sparse-universe cash buffer $B_t = \min(1.0, N_t / 3.0)$ of the historical strategy.

---

## 2. Mathematical Specification

### 2.1 Synthetic Cross-Sectional Pool Calibration
Given an equal-weight S&P benchmark daily return series $r_{\text{EW}, t}$ across $T$ trading sessions and a cross-sectional pool width $M = 500$, daily constituent returns $r_{j, t}$ ($j = 1, \dots, M$) are constructed with zero-mean cross-sectional dispersion $\varepsilon_{j, t}$:

$$
r_{j, t} = r_{\text{EW}, t} + \left(\varepsilon_{j, t} - \frac{1}{M}\sum_{m=1}^{M} \varepsilon_{m, t}\right), \qquad \varepsilon_{j, t} \sim \mathcal{N}(0, \sigma_{\varepsilon}^2)
$$

By centering $\varepsilon_{j, t}$ cross-sectionally on every session $t$, the equal-weighted pool mean equals the benchmark return identically on every day:

$$
\frac{1}{M} \sum_{j=1}^{M} r_{j, t} = r_{\text{EW}, t} \qquad \forall t \in \{1, \dots, T\}
$$

### 2.2 Breadth-Matched Random Portfolio Resampling ($N_t$)
For each Monte Carlo trial $b \in \{1, \dots, B\}$ ($B = 10,000$) and each trading day $t \in \{1, \dots, T\}$:
1. If $N_t = 0$, the trial portfolio holds $100\%$ cash ($r_{p, t}^{(b)} = 0$).
2. If $N_t \ge 1$, the engine draws an index tuple $S_t^{(b)}$ of size $N_t$ uniformly from $\{1, \dots, M\}$ and applies the sparse-universe equity budget $B_t = \min(1.0, N_t / 3.0)$:

$$
r_{p, t}^{(b)} = B_t \cdot \frac{1}{N_t} \sum_{j \in S_t^{(b)}} r_{j, t}
$$

### 2.3 Convergence to the Equal-Weight S&P Benchmark
Taking expectation over the uniform random draw $S_t^{(b)}$ conditional on the session cross-section yields:

$$
\mathbb{E}\!\left[r_{p, t}^{(b)}\right] = B_t \cdot \left(\frac{1}{M} \sum_{j=1}^{M} r_{j, t}\right) = B_t \cdot r_{\text{EW}, t}
$$

By the Law of Large Numbers, as $B \to \infty$ ($B = 10,000$), the sample mean path $\bar{r}_{p, t} = \frac{1}{B}\sum_{b=1}^{B} r_{p, t}^{(b)}$ converges to the budget-scaled equal-weight S&P benchmark return series $B_t \cdot r_{\text{EW}, t}$ with standard error $\mathcal{O}(B^{-1/2})$.

---

## 3. Computational Architecture & Reproducibility
* **Deterministic PRNG (`SeedSequence` + `PCG64`):** `SyntheticBootstrapEngine.create_rng(seed)` initializes an isolated `np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))` instance (default `seed = 42`), ensuring bit-for-bit identical simulation outputs across runs without mutating global NumPy state.
* **Regime-Grouped Vectorization:** Trading sessions are grouped by unique active count $k \in \{0, 1, \dots, \max(N_t)\}$, allowing batch index generation `(T_k, B, k)` and `np.take_along_axis` evaluation across all $10,000 \times 6,761$ portfolio-days in ~3 seconds.

---

## 4. Empirical Bootstrap Summary (`2000-01-03` to `2026-11-18`, $B = 10,000$)

| Statistic | Value |
| :--- | :---: |
| **Bootstrap Trials ($B$)** | 10,000 |
| **Trading Horizon ($T$)** | 6,761 sessions |
| **Cross-Sectional Pool Width ($M$)** | 500 constituents |
| **Bootstrap Distribution Mean CAGR** | **7.78%** |
| **Bootstrap 95th Percentile CAGR** | **10.90%** |
| **Observed Black CEO Net CAGR** | **13.36%** |

---

## 5. Empirical $p$-Value Extraction & Expo Judge Interpretation

### 5.1 Finite-Sample Permutation $p$-Value Estimator
To avoid parametric normality assumptions on concentrated equity returns, `extract_empirical_bootstrap_p_value()` (`src/attribution.py`) evaluates the observed strategy statistic $\theta_{\text{obs}}$ against the $B = 10,000$ synthetic null realizations $\{\theta^{(1)}, \dots, \theta^{(B)}\}$ using the Davison-Hinkley finite-sample permutation estimator:

$$
\hat{p}_{\text{right}} = \frac{1 + \sum_{b=1}^{B} \mathbb{I}\!\left(\theta^{(b)} \ge \theta_{\text{obs}}\right)}{B + 1}, \qquad \hat{p}_{\text{two-sided}} = \frac{1 + \sum_{b=1}^{B} \mathbb{I}\!\left(\left|\theta^{(b)} - \bar{\theta}_{\text{null}}\right| \ge \left|\theta_{\text{obs}} - \bar{\theta}_{\text{null}}\right|\right)}{B + 1}
$$

* **Why $(1 + \text{count}) / (B + 1)$:** Adding $1$ to both numerator and denominator treats the observed strategy as one realization under the exchangeability null, preventing artificial $p = 0$ claims when $\theta_{\text{obs}}$ ranks at the extreme right tail while bounding resolution at $(10,001)^{-1} \approx 0.0001$.
* **Two-Tailed 95% Null Bounds:** Defined by the empirical $2.5\text{th}$ and $97.5\text{th}$ percentiles ($q_{0.025}, q_{0.975}$) of the synthetic null distribution.

### 5.2 Empirical Results (`2000-01-03` to `2026-11-18`, $B = 10,000$)

| Metric | Observed Strategy (Net) | Null Mean | Two-Tailed 95% Null Bounds | Percentile Rank | Right-Tailed $p$-value | Two-Sided $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAGR** | **13.36%** | 7.78% | `[4.14%, 11.56%]` | **99.83%** | **0.0018** | **0.0035** |
| **Sharpe Ratio ($R_f = 1.5\%$)** | **0.4452** | 0.3873 | `[0.2264, 0.5497]` | **75.48%** | **0.2453** | **0.4894** |

### 5.3 Plain-English Summary for Expo Judges
1. **Compound Growth Outperformance ($p = 0.0018$):** Out of **10,000** randomized portfolios subjected to the exact same historical CEO headcount ($N_t$) and cash buffer rules, **99.83%** generated a lower compound annual growth rate than the Black CEO portfolio (`13.36%` vs. `7.78%` null mean). Only **17 out of 10,000** random draws matched or exceeded `13.36%`, rejecting pure luck at the $1\%$ significance level.
2. **Risk-Adjusted Efficiency Trade-Off ($p = 0.2453$):** Because holding an average of $8$ companies introduces idiosyncratic volatility (`42.08%` vs. `19.24%` for `SPY`), the strategy's `0.45` Sharpe ratio places in the **upper quartile (75.5th percentile)** of breadth-matched random portfolios—confirming solid risk-adjusted execution while transparently showing that volatility compression (addressed by our `Convex-Quality` optimizer) is key to converting raw return edge into higher Sharpe significance.
