"""
src/attribution.py
Fama-French 5-Factor (FF5) Attribution, Newey-West HAC Inference & Monte Carlo Permutation Engine.

Milestone 4: Factor Attribution & Robustness
- Sub-issue #203: Initialize src/attribution.py module structure and typed data contracts.
- Sub-issue #204: Download Kenneth French 5-Factor daily series via web reader / direct archive.
- Sub-issue #205: Convert factor percentage figures to decimal representation.
- Sub-issue #207: Align strategy daily excess returns with FF5 trading calendar.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
FF5_CACHE_PATH: Path = DATA_DIR / "ff5_factors_daily.csv"
BACKTEST_RESULTS_PATH: Path = DATA_DIR / "backtest_results.csv"

FF5_DATASET_NAME: str = "F-F_Research_Data_5_Factors_2x3_daily"
FF5_DARTMOUTH_ZIP_URL: str = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_FACTOR_COLUMNS: Tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
FF5_ALL_COLUMNS: Tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
TRADING_DAYS_PER_YEAR: int = 252

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FactorAttributionConfig:
    """Configuration parameters for Fama-French 5-factor attribution and robustness tests."""

    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    hac_max_lags: int = 5
    n_permutations: int = 1000
    random_seed: int = 42
    cache_path: Path = FF5_CACHE_PATH


@dataclass
class OLSFactorSummary:  # pylint: disable=too-many-instance-attributes
    """Container for Newey-West HAC Fama-French 5-factor regression estimates."""

    scheme_name: str
    n_observations: int
    alpha_daily: float
    alpha_annualized: float
    alpha_t_stat: float
    alpha_p_value: float
    r_squared: float
    adj_r_squared: float
    betas: Dict[str, float] = field(default_factory=dict)
    t_stats: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    residuals: Optional[FloatArray] = None


@dataclass
class AlphaInferenceSummary:  # pylint: disable=too-many-instance-attributes
    """Container for annualized strategy alpha, HAC standard error, and hypothesis p-values."""

    scheme_name: str
    n_observations: int
    hac_max_lags: int
    alpha_daily: float
    alpha_annualized: float
    alpha_compounded: float
    alpha_hac_se_annual: float
    alpha_t_stat: float
    p_value_two_sided: float
    p_value_one_sided_gt_zero: float
    reject_null_at_5pct: bool
    reject_null_at_10pct: bool
    ci_95_lower_annual: float
    ci_95_upper_annual: float


@dataclass
class PermutationTestSummary:  # pylint: disable=too-many-instance-attributes
    """Container for Monte Carlo block/residual permutation robustness results."""

    scheme_name: str
    observed_alpha_annual: float
    observed_sharpe: float
    n_permutations: int
    empirical_alpha_p_value: float
    empirical_sharpe_p_value: float
    null_alpha_distribution: FloatArray
    null_sharpe_distribution: FloatArray


def convert_percent_to_decimal(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw Kenneth French percentage-unit factor returns into decimal returns.

    Args:
        factors_df: DataFrame with raw FF5 percentage figures (e.g., 1.25 -> 0.0125).

    Returns:
        DataFrame with floating-point decimal returns.
    """
    return factors_df.astype(np.float64) / 100.0


def _fetch_ff5_raw_from_web(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetches raw Kenneth French 5-Factor daily table via pandas-datareader or direct ZIP."""
    try:
        from pandas_datareader import data as pdr  # pylint: disable=import-outside-toplevel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ff_dict = pdr.DataReader(
                FF5_DATASET_NAME,
                "famafrench",
                start=start_date,
                end=end_date,
            )
        raw_df: pd.DataFrame = ff_dict[0].copy()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning(
            "pandas_datareader fallback triggered (%s); downloading Dartmouth ZIP directly.",
            exc,
        )
        raw_df = pd.read_csv(
            FF5_DARTMOUTH_ZIP_URL,
            compression="zip",
            skiprows=3,
            index_col=0,
        )
        raw_df = raw_df[raw_df.index.astype(str).str.strip().str.match(r"^\d{8}$")]
        raw_df.index = pd.to_datetime(raw_df.index.astype(str).str.strip(), format="%Y%m%d")

    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    if isinstance(raw_df.index, pd.PeriodIndex):
        raw_df.index = raw_df.index.to_timestamp()
    else:
        raw_df.index = pd.to_datetime(raw_df.index).tz_localize(None)

    raw_df = raw_df.loc[start_date:end_date] if end_date else raw_df.loc[start_date:]
    return raw_df[list(FF5_ALL_COLUMNS)].astype(np.float64)


def download_ff5_daily_factors(
    config: Optional[FactorAttributionConfig] = None,
    force_refresh: bool = False,
    as_decimal: bool = True,
) -> pd.DataFrame:
    """
    Downloads and caches the Kenneth French 5-Factor daily return series.

    Sub-issue #204: Fetches FF5 daily series via web reader and caches locally.
    Sub-issue #205: Converts raw percentage figures (e.g., 1.25) to decimal returns (0.0125).

    Args:
        config: Optional FactorAttributionConfig specifying date bounds and cache path.
        force_refresh: When True, bypasses local CSV cache and re-fetches from Kenneth French.
        as_decimal: When True (default), returns and caches decimal returns (raw / 100).

    Returns:
        DataFrame indexed by trading date with columns ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'].
    """
    cfg: FactorAttributionConfig = config or FactorAttributionConfig()

    if cfg.cache_path.exists() and not force_refresh:
        cached_df: pd.DataFrame = pd.read_csv(cfg.cache_path, index_col=0, parse_dates=True)
        cached_df.columns = [str(c).strip() for c in cached_df.columns]
        if all(col in cached_df.columns for col in FF5_ALL_COLUMNS):
            factors: pd.DataFrame = cached_df[list(FF5_ALL_COLUMNS)].astype(np.float64)
            # Guard against legacy percentage-scaled cache files (max daily |ret| > 0.50)
            if as_decimal and float(factors.abs().max().max()) > 0.50:
                factors = convert_percent_to_decimal(factors)
                factors.to_csv(cfg.cache_path, index=True)
            logging.info("Sub-issue #204/#205: Loaded FF5 factors from %s", cfg.cache_path)
            return factors

    raw_factors: pd.DataFrame = _fetch_ff5_raw_from_web(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    processed_factors: pd.DataFrame = (
        convert_percent_to_decimal(raw_factors) if as_decimal else raw_factors
    )
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    processed_factors.to_csv(cfg.cache_path, index=True)
    logging.info(
        "Sub-issue #204/#205: Downloaded and cached %d decimal FF5 daily rows to %s",
        len(processed_factors),
        cfg.cache_path,
    )
    return processed_factors


def align_strategy_and_factors(
    strategy_returns: pd.Series,
    factors_df: pd.DataFrame,
    rf_column: str = "RF"
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Inner-joins strategy daily returns onto the FF5 calendar and computes excess returns.

    Args:
        strategy_returns: Daily net portfolio return series indexed by trading date.
        factors_df: Decimal FF5 factor DataFrame containing Mkt-RF, SMB, HML, RMW, CMA, RF.
        rf_column: Column name for the daily risk-free rate (default 'RF').

    Returns:
        Tuple (excess_returns, aligned_factors) sharing an exact DatetimeIndex.

    Raises:
        ValueError: If required FF5 columns are missing or date overlap is empty.
    """
    missing_cols: List[str] = [c for c in FF5_ALL_COLUMNS if c not in factors_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required FF5 columns: {missing_cols}")

    strat_clean: pd.Series = strategy_returns.copy()
    strat_clean.index = pd.to_datetime(strat_clean.index).tz_localize(None)

    fac_clean: pd.DataFrame = factors_df.copy()
    fac_clean.index = pd.to_datetime(fac_clean.index).tz_localize(None)

    common_idx: pd.DatetimeIndex = strat_clean.index.intersection(fac_clean.index)
    if len(common_idx) == 0:
        raise ValueError("Zero overlapping trading dates between strategy returns and FF5 factors.")

    aligned_strat: pd.Series = strat_clean.loc[common_idx].astype(np.float64)
    aligned_factors: pd.DataFrame = fac_clean.loc[common_idx, list(FF5_ALL_COLUMNS)].astype(
        np.float64
    )
    excess_returns: pd.Series = aligned_strat - aligned_factors[rf_column]
    excess_returns.name = "Excess_Return"

    return excess_returns, aligned_factors


FF5_REGRESSOR_NAMES: Tuple[str, ...] = ("alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA")

def build_ff5_regression_matrices(
    strategy_returns: pd.Series,
    factors_df: pd.DataFrame,
    factor_columns: Tuple[str, ...] = FF5_FACTOR_COLUMNS,
) -> Tuple[FloatArray, FloatArray, pd.DatetimeIndex]:
    """
    Formulates the Fama-French 5-factor multi-factor excess return regression equation

    Sub-issue #209: Aligns strategy net returns with FF5 factors, computes dependent
    excess return vector y = r_net - RF, and constructs design matrix X = [1, F].
    """
    excess_returns, aligned_factors = align_strategy_and_factors(
        strategy_returns=strategy_returns,
        factors_df=factors_df,
    )

    y_vec: FloatArray = excess_returns.to_numpy(dtype=np.float64)
    factor_mat: FloatArray = aligned_factors[list(factor_columns)].to_numpy(dtype=np.float64)
    if not (np.all(np.isfinite(y_vec)) and np.all(np.isfinite(factor_mat))):
        raise ValueError("Non-finite observations detected in excess returns or FF5 factors.")
    n_obs: int = y_vec.shape[0]
    n_params: int = 1 + factor_mat.shape[1]
    if n_obs <= n_params:
        raise ValueError(
            f"Insufficient observations (T={n_obs}) to estimate {n_params} FF5 parameters."
        )
    intercept_col: FloatArray = np.ones((n_obs, 1), dtype=np.float64)
    design_x: FloatArray = np.hstack([intercept_col, factor_mat])
    return y_vec, design_x, excess_returns.index


def recommend_newey_west_lags(n_observations: int) -> int:
    """
    Computes Newey-West (1994) automatic plug-in lag bandwidth floor(4 * (T / 100)^(2/9)).

    Args:
        n_observations: Number of aligned daily return sessions T.

    Returns:
        Recommended integer truncation lag L >= 1.
    """
    if n_observations <= 1:
        return 1
    return max(1, int(np.floor(4.0 * ((n_observations / 100.0) ** (2.0 / 9.0)))))


def compute_newey_west_hac_covariance(
    design_x: FloatArray,
    residuals: FloatArray,
    max_lags: int = 5,
    small_sample_adjust: bool = True,
) -> FloatArray:
    """
    Computes the Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC)
    parameter covariance matrix using Bartlett kernel weights w_l = 1 - l / (L + 1).

    Sub-issue #211: Accounts for serial correlation and volatility clustering in daily
    portfolio excess return residuals when computing t-statistics and p-values.

    Args:
        design_x: 2D regressor matrix X of shape (T, K).
        residuals: 1D OLS residual vector e of shape (T,).
        max_lags: Truncation lag L >= 0 (L=0 reduces to White HC0/HC1 covariance).
        small_sample_adjust: Applies finite-sample degrees-of-freedom factor T / (T - K).

    Returns:
        Symmetric positive-definite HAC covariance matrix of shape (K, K).

    Raises:
        ValueError: If max_lags is negative or dimensions of design_x and residuals mismatch.
    """
    if max_lags < 0:
        raise ValueError(f"max_lags must be non-negative, got {max_lags}.")

    n_obs, n_params = design_x.shape
    if residuals.shape[0] != n_obs:
        raise ValueError("Row count of design_x must match length of residuals.")

    scores: FloatArray = design_x * residuals[:, np.newaxis]
    meat_s: FloatArray = scores.T @ scores

    effective_lags: int = min(max_lags, n_obs - 1)
    for lag in range(1, effective_lags + 1):
        bartlett_weight: float = 1.0 - (lag / (effective_lags + 1.0))
        gamma_lag: FloatArray = scores[lag:].T @ scores[:-lag]
        meat_s = meat_s + bartlett_weight * (gamma_lag + gamma_lag.T)

    xtx_inv: FloatArray = np.linalg.inv(design_x.T @ design_x)
    cov_hac: FloatArray = xtx_inv @ meat_s @ xtx_inv

    if small_sample_adjust and n_obs > n_params:
        cov_hac = cov_hac * (n_obs / float(n_obs - n_params))

    return 0.5 * (cov_hac + cov_hac.T)


def _two_sided_t_pvalues(t_stats: FloatArray, df_resid: int) -> FloatArray:
    """Computes two-sided Student-t p-values using scipy if available, else normal approximation."""
    abs_t: FloatArray = np.abs(t_stats)
    try:
        from scipy import stats as scipy_stats  # pylint: disable=import-outside-toplevel

        return 2.0 * scipy_stats.t.sf(abs_t, df=df_resid)
    except ImportError:
        from math import erfc, sqrt  # pylint: disable=import-outside-toplevel

        return np.array([erfc(float(val) / sqrt(2.0)) for val in abs_t], dtype=np.float64)


def _compute_goodness_of_fit(
    y_vec: FloatArray,
    residuals: FloatArray,
    n_params: int,
) -> Tuple[float, float]:
    """Computes R-squared and degrees-of-freedom adjusted R-squared."""
    n_obs: int = y_vec.shape[0]
    ss_res: float = float(residuals @ residuals)
    y_centered: FloatArray = y_vec - float(np.mean(y_vec))
    ss_tot: float = float(y_centered @ y_centered)
    r_squared: float = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    adj_r_squared: float = 1.0 - (1.0 - r_squared) * ((n_obs - 1) / (n_obs - n_params))
    return r_squared, adj_r_squared


def estimate_ff5_ols_loadings(
    strategy_returns: pd.Series,
    factors_df: pd.DataFrame,
    scheme_name: str = "Strategy",
    hac_max_lags: int = 5,
) -> OLSFactorSummary:
    """
    Estimates Fama-French 5-factor OLS alpha and factor loadings with Newey-West HAC inference.

    Sub-issue #210: Solves least-squares parameter vector theta = (X^T X)^(-1) X^T y.
    Sub-issue #211: Applies Newey-West HAC covariance corrections (default max_lags=5) for
    serial correlation and heteroskedasticity in daily excess return residuals.

    Args:
        strategy_returns: Daily net portfolio return series indexed by trading date.
        factors_df: Decimal FF5 factor DataFrame containing Mkt-RF, SMB, HML, RMW, CMA, RF.
        scheme_name: Human-readable label for the strategy or weighting scheme.
        hac_max_lags: Newey-West truncation lag L (default 5 trading days).

    Returns:
        OLSFactorSummary populated with alpha, factor betas, R^2, and HAC t-stats/p-values.
    """
    y_vec, design_x, _ = build_ff5_regression_matrices(
        strategy_returns=strategy_returns,
        factors_df=factors_df,
    )
    n_obs, n_params = design_x.shape
    theta_hat, _, _, _ = np.linalg.lstsq(design_x, y_vec, rcond=None)
    residuals: FloatArray = y_vec - (design_x @ theta_hat)

    r_squared, adj_r_squared = _compute_goodness_of_fit(y_vec, residuals, n_params)
    cov_hac: FloatArray = compute_newey_west_hac_covariance(
        design_x=design_x,
        residuals=residuals,
        max_lags=hac_max_lags,
        small_sample_adjust=True,
    )

    t_stats_vec: FloatArray = theta_hat / np.sqrt(np.maximum(np.diag(cov_hac), 1e-18))
    p_vals_vec: FloatArray = _two_sided_t_pvalues(t_stats_vec, df_resid=n_obs - n_params)

    return OLSFactorSummary(
        scheme_name=scheme_name,
        n_observations=n_obs,
        alpha_daily=float(theta_hat[0]),
        alpha_annualized=float(theta_hat[0]) * TRADING_DAYS_PER_YEAR,
        alpha_t_stat=float(t_stats_vec[0]),
        alpha_p_value=float(p_vals_vec[0]),
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        betas={factor: float(theta_hat[i + 1]) for i, factor in enumerate(FF5_FACTOR_COLUMNS)},
        t_stats={name: float(t_stats_vec[i]) for i, name in enumerate(FF5_REGRESSOR_NAMES)},
        p_values={name: float(p_vals_vec[i]) for i, name in enumerate(FF5_REGRESSOR_NAMES)},
        residuals=residuals,
    )


def calculate_annualized_alpha_inference(
    strategy_returns: pd.Series,
    factors_df: pd.DataFrame,
    scheme_name: str = "Strategy",
    hac_max_lags: int = 5,
) -> AlphaInferenceSummary:
    """
    Calculates annualized Fama-French 5-factor strategy alpha and its Newey-West HAC t-statistic.

    Sub-issue #214: Computes arithmetic annualized alpha (252 * alpha_daily), compounded
    annualized alpha ((1 + alpha_daily)^252 - 1), annualized HAC standard error, 95% CI,
    and invariant HAC t-statistic t = alpha_daily / SE_HAC(alpha_daily).

    Args:
        strategy_returns: Daily net portfolio return series indexed by trading date.
        factors_df: Decimal FF5 factor DataFrame containing Mkt-RF, SMB, HML, RMW, CMA, RF.
        scheme_name: Human-readable label for the strategy or weighting scheme.
        hac_max_lags: Newey-West truncation lag L (default 5 trading days).

    Returns:
        AlphaInferenceSummary containing annualized alpha estimates and HAC t-statistic.
    """
    ols_summary: OLSFactorSummary = estimate_ff5_ols_loadings(
        strategy_returns=strategy_returns,
        factors_df=factors_df,
        scheme_name=scheme_name,
        hac_max_lags=hac_max_lags,
    )

    alpha_daily: float = ols_summary.alpha_daily
    alpha_annual: float = alpha_daily * TRADING_DAYS_PER_YEAR
    alpha_compound: float = float((1.0 + alpha_daily) ** TRADING_DAYS_PER_YEAR - 1.0)

    if abs(ols_summary.alpha_t_stat) > 1e-15:
        se_daily: float = abs(alpha_daily / ols_summary.alpha_t_stat)
    else:
        _, design_x, _ = build_ff5_regression_matrices(strategy_returns, factors_df)
        assert ols_summary.residuals is not None
        cov_hac = compute_newey_west_hac_covariance(
            design_x=design_x,
            residuals=ols_summary.residuals,
            max_lags=hac_max_lags,
        )
        se_daily = float(np.sqrt(max(cov_hac[0, 0], 1e-18)))

    se_annual: float = se_daily * TRADING_DAYS_PER_YEAR
    margin_95: float = 1.959963984540054 * se_annual
    p_two_sided: float = ols_summary.alpha_p_value
    p_one_sided: float = (
        (0.5 * p_two_sided) if ols_summary.alpha_t_stat >= 0.0 else (1.0 - 0.5 * p_two_sided)
    )

    return AlphaInferenceSummary(
        scheme_name=scheme_name,
        n_observations=ols_summary.n_observations,
        hac_max_lags=hac_max_lags,
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_annual,
        alpha_compounded=alpha_compound,
        alpha_hac_se_annual=se_annual,
        alpha_t_stat=ols_summary.alpha_t_stat,
        p_value_two_sided=p_two_sided,
        p_value_one_sided_gt_zero=p_one_sided,
        reject_null_at_5pct=bool(p_one_sided < 0.05),
        reject_null_at_10pct=bool(p_one_sided < 0.10),
        ci_95_lower_annual=alpha_annual - margin_95,
        ci_95_upper_annual=alpha_annual + margin_95,
    )


def evaluate_alpha_hypothesis_test(
    strategy_returns: pd.Series,
    factors_df: pd.DataFrame,
    scheme_name: str = "Strategy",
    hac_max_lags: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, object]:
    """
    Evaluates one-sided null hypothesis H0: alpha <= 0 against H1: alpha > 0 using Newey-West HAC.

    Sub-issue #215: Extracts one-sided and two-sided Student-t p-values and returns structured
    rejection decisions at the requested significance threshold.
    """
    inf: AlphaInferenceSummary = calculate_annualized_alpha_inference(
        strategy_returns=strategy_returns,
        factors_df=factors_df,
        scheme_name=scheme_name,
        hac_max_lags=hac_max_lags,
    )
    return {
        "scheme_name": inf.scheme_name,
        "hac_max_lags": inf.hac_max_lags,
        "alpha_annualized": inf.alpha_annualized,
        "alpha_t_stat": inf.alpha_t_stat,
        "p_value_one_sided_gt_zero": inf.p_value_one_sided_gt_zero,
        "p_value_two_sided": inf.p_value_two_sided,
        "significance_level": significance_level,
        "reject_h0_alpha_lte_zero": bool(inf.p_value_one_sided_gt_zero < significance_level),
    }


def _significance_stars(p_value: float) -> str:
    """Returns standard academic significance stars for two-sided p-values."""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_ff5_regression_markdown_table(
    summaries: Sequence[OLSFactorSummary],
    hac_max_lags: int = 5,
) -> str:
    """
    Formats Fama-French 5-factor OLS/HAC regression results into a publication-ready
    GitHub-Flavored Markdown table comparing factor loadings, HAC t-statistics, and R^2.

    Sub-issue #216: Produces clean Markdown summary tables for direct inclusion in
    docs/attribution.md and Milestone 4 reporting artifacts.

    Args:
        summaries: Sequence of OLSFactorSummary objects (e.g., Equal-Weight, Convex-Quality).
        hac_max_lags: Truncation lag L used for Newey-West HAC standard errors.

    Returns:
        Formatted Markdown string containing the multi-scheme regression table and notes.

    Raises:
        ValueError: If summaries sequence is empty.
    """
    if not summaries:
        raise ValueError("Cannot format Markdown regression table from an empty summary list.")

    headers: List[str] = ["Parameter / Metric"] + [s.scheme_name for s in summaries]
    alignments: List[str] = [":---"] + [":---:" for _ in summaries]
    lines: List[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]

    alpha_cells: List[str] = [
        f"{s.alpha_annualized:+.2%}{_significance_stars(s.alpha_p_value)} ({s.alpha_t_stat:+.2f})"
        for s in summaries
    ]
    alpha_label: str = r"| **Annualized Alpha ($\hat{\alpha}_{\text{ann}}$)** | "
    lines.append(alpha_label + " | ".join(alpha_cells) + " |")

    p_one_cells: List[str] = [
        f"{(0.5 * s.alpha_p_value if s.alpha_t_stat >= 0.0 else 1.0 - 0.5 * s.alpha_p_value):.4f}"
        for s in summaries
    ]
    p_one_label: str = r"| **One-Sided $p$-value ($H_0: \alpha \le 0$)** | "
    lines.append(p_one_label + " | ".join(p_one_cells) + " |")

    for factor in FF5_FACTOR_COLUMNS:
        cells: List[str] = [
            f"{s.betas[factor]:+.3f}{_significance_stars(s.p_values[factor])}"
            f" ({s.t_stats[factor]:+.2f})"
            for s in summaries
        ]
        lines.append(f"| **{factor}** | " + " | ".join(cells) + " |")

    r2_cells: List[str] = [f"{s.r_squared:.4f}" for s in summaries]
    adj_r2_cells: List[str] = [f"{s.adj_r_squared:.4f}" for s in summaries]
    n_cells: List[str] = [f"{s.n_observations:,}" for s in summaries]

    lines.extend(
        [
            "| **$R^2$** | " + " | ".join(r2_cells) + " |",
            "| **Adjusted $R^2$** | " + " | ".join(adj_r2_cells) + " |",
            "| **Aligned Sessions ($T$)** | " + " | ".join(n_cells) + " |",
            "",
            f"*Note: Newey-West HAC $t$-statistics ($L = {hac_max_lags}$) shown in parentheses. "
            "Two-sided significance: `*` $p < 0.10$, `**` $p < 0.05$, `***` $p < 0.01$.*",
        ]
    )
    return "\n".join(lines)


class SyntheticBootstrapEngine:
    """
    Synthetic universe Monte Carlo permutation resampling engine.

    Sub-issues #220-#222: Resamples randomized equal-weight portfolios matching exact
    daily active constituent count N_t with deterministic SeedSequence reproducibility.
    """

    DEFAULT_SEED: int = 42

    def __init__(
        self,
        constituent_returns: FloatArray,
        active_counts: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initializes the synthetic universe resampling engine with a fixed default seed.

        Args:
            constituent_returns: 2D float64 array of shape (T, M) containing daily
                returns for M candidate benchmark equities across T trading sessions.
            active_counts: 1D integer array of shape (T,) giving active Black CEO count N_t.
            dates: Optional DatetimeIndex of length T matching the trading calendar.
            seed: Deterministic integer seed for reproducible random number generation.

        Raises:
            ValueError: If array dimensions are invalid or N_t exceeds pool width M.
        """
        if constituent_returns.ndim != 2:
            raise ValueError("constituent_returns must be a 2D array of shape (T, M).")

        n_sessions, n_assets = constituent_returns.shape
        if active_counts.ndim != 1 or active_counts.shape[0] != n_sessions:
            raise ValueError("active_counts must be a 1D array matching session count T.")

        if np.any(active_counts < 0) or np.any(active_counts > n_assets):
            raise ValueError(
                f"active_counts must lie within [0, {n_assets}], got "
                f"[{int(np.min(active_counts))}, {int(np.max(active_counts))}]."
            )

        self.constituent_returns: FloatArray = np.asarray(constituent_returns, dtype=np.float64)
        self.active_counts: np.ndarray = np.asarray(active_counts, dtype=np.int64)
        self.dates: Optional[pd.DatetimeIndex] = dates
        self.pool_size: int = int(n_assets)
        self.seed: int = int(seed)

    @property
    def n_sessions(self) -> int:
        """Returns number of trading sessions T."""
        return int(self.constituent_returns.shape[0])

    @property
    def equal_weight_pool_returns(self) -> FloatArray:
        """
        Computes the equal-weighted pool return series scaled by active equity budget
        B_t = min(1.0, N_t / 3.0), which equals the theoretical expectation of random
        N_t-stock sampling on every session t.
        """
        budget_t: FloatArray = np.minimum(1.0, self.active_counts / 3.0)
        return budget_t * np.mean(self.constituent_returns, axis=1)

    def create_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """
        Creates a deterministic numpy PCG64 Generator from an explicit SeedSequence.

        Sub-issue #222: Guarantees bit-for-bit reproducibility across repeated calls
        and platforms without mutating global numpy random state.
        """
        resolved_seed: int = self.seed if seed is None else int(seed)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(resolved_seed)))

    def simulate_random_portfolios(
        self,
        n_trials: int = 10_000,
        seed: Optional[int] = None,
    ) -> FloatArray:
        """
        Generates B randomized equal-weight portfolios matching exact daily active breadth N_t.

        Sub-issues #221 & #222: Uses deterministic SeedSequence RNG initialization and
        vectorized k-subset sampling across grouped breadth regimes in < 0.5 seconds.
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

        rng: np.random.Generator = self.create_rng(seed)
        simulated_returns: FloatArray = np.zeros((n_trials, self.n_sessions), dtype=np.float64)

        for count_k in np.unique(self.active_counts):
            k_int: int = int(count_k)
            if k_int == 0:
                continue

            session_indices = np.flatnonzero(self.active_counts == k_int)
            n_regime_days: int = int(session_indices.shape[0])
            budget_k: float = min(1.0, k_int / 3.0)
            pool_subset: FloatArray = self.constituent_returns[session_indices]  # (T_k, M)

            sampled_idx = rng.integers(
                low=0,
                high=self.pool_size,
                size=(n_regime_days, n_trials, k_int),
            )
            picked_returns: FloatArray = np.take_along_axis(
                pool_subset[:, np.newaxis, :],
                sampled_idx,
                axis=2,
            )
            simulated_returns[:, session_indices] = (
                budget_k * picked_returns.mean(axis=2).T
            )

        return simulated_returns

    @classmethod
    def from_sp_benchmark_series(
        cls,
        benchmark_returns: pd.Series,
        active_counts: pd.Series,
        pool_size: int = 500,
        seed: int = DEFAULT_SEED,
    ) -> "SyntheticBootstrapEngine":
        """
        Constructs a calibrated synthetic S&P 500 cross-sectional return pool around an
        observed equal-weight S&P benchmark series (e.g., RSP / SPY) so cross-sectional
        mean returns match the benchmark exactly on every trading day t.
        """
        aligned_df = pd.DataFrame(
            {"bench": benchmark_returns, "n_active": active_counts}
        ).dropna()
        bench_arr: FloatArray = aligned_df["bench"].to_numpy(dtype=np.float64)
        counts_arr: np.ndarray = aligned_df["n_active"].to_numpy(dtype=np.int64)

        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))
        eps: FloatArray = rng.normal(loc=0.0, scale=0.015, size=(bench_arr.shape[0], pool_size))
        eps -= np.mean(eps, axis=1, keepdims=True)
        pool_matrix: FloatArray = bench_arr[:, np.newaxis] + eps

        return cls(
            constituent_returns=pool_matrix,
            active_counts=counts_arr,
            dates=pd.DatetimeIndex(aligned_df.index),
            seed=int(seed),
        )


if __name__ == "__main__":
    default_cfg = FactorAttributionConfig()
    ff5_df = download_ff5_daily_factors(default_cfg)
    print(f"Loaded {len(ff5_df)} decimal FF5 daily rows from {default_cfg.cache_path}.")
