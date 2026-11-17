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
from typing import Dict, List, Optional, Tuple
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


def _two_sided_t_pvalues(t_stats: FloatArray, df_resid: int) -> FloatArray:
    """Computes two-sided Student-t p-values using scipy if available, else normal approximation."""
    abs_t: FloatArray = np.abs(t_stats)
    try:
        from scipy import stats as scipy_stats  # pylint: disable=import-outside-toplevel

        return 2.0 * scipy_stats.t.sf(abs_t, df=df_resid)
    except ImportError:
        from math import erfc, sqrt  # pylint: disable=import-outside-toplevel

        return np.array([erfc(float(val) / sqrt(2.0)) for val in abs_t], dtype=np.float64)


def estimate_ff5_ols_loadings(
    strategy_returns: pd.Series,
    factors_df: pd.DataFrame,
    scheme_name: str = "Strategy",
) -> OLSFactorSummary:
    """
    Estimates Fama-French 5-factor OLS alpha and factor loadings (Mkt-RF, SMB, HML, RMW, CMA).

    Sub-issue #210: Solves least-squares parameter vector theta = (X^T X)^(-1) X^T y via
    numerical QR/SVD least squares, extracting daily/annualized alpha, factor betas, R^2,
    and adjusted R^2.
    """
    y_vec, design_x, _ = build_ff5_regression_matrices(
        strategy_returns=strategy_returns,
        factors_df=factors_df,
    )
    n_obs, n_params = design_x.shape
    df_resid: int = n_obs - n_params
    theta_hat, _, _, _ = np.linalg.lstsq(design_x, y_vec, rcond=None)
    fitted: FloatArray = design_x @ theta_hat
    residuals: FloatArray = y_vec - fitted
    ss_res: float = float(residuals @ residuals)
    y_centered: FloatArray = y_vec - float(np.mean(y_vec))
    ss_tot: float = float(y_centered @ y_centered)
    r_squared: float = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    adj_r_squared: float = 1.0 - (1.0 - r_squared) * ((n_obs - 1) / df_resid)
    sigma2_hat: float = ss_res / df_resid
    xtx_inv: FloatArray = np.linalg.inv(design_x.T @ design_x)
    cov_ols: FloatArray = sigma2_hat * xtx_inv
    std_errors: FloatArray = np.sqrt(np.maximum(np.diag(cov_ols), 1e-18))
    t_stats_vec: FloatArray = theta_hat / std_errors
    p_vals_vec: FloatArray = _two_sided_t_pvalues(t_stats_vec, df_resid=df_resid)
    alpha_daily: float = float(theta_hat[0])
    betas_dict: Dict[str, float] = {
        factor: float(theta_hat[idx + 1]) for idx, factor in enumerate(FF5_FACTOR_COLUMNS)
    }
    t_stats_dict: Dict[str, float] = {
        name: float(t_stats_vec[idx]) for idx, name in enumerate(FF5_REGRESSOR_NAMES)
    }
    p_values_dict: Dict[str, float] = {
        name: float(p_vals_vec[idx]) for idx, name in enumerate(FF5_REGRESSOR_NAMES)
    }
    return OLSFactorSummary(
        scheme_name=scheme_name,
        n_observations=n_obs,
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_daily * TRADING_DAYS_PER_YEAR,
        alpha_t_stat=float(t_stats_vec[0]),
        alpha_p_value=float(p_vals_vec[0]),
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        betas=betas_dict,
        t_stats=t_stats_dict,
        p_values=p_values_dict,
        residuals=residuals,
    )

if __name__ == "__main__":
    default_cfg = FactorAttributionConfig()
    ff5_df = download_ff5_daily_factors(default_cfg)
    print(f"Loaded {len(ff5_df)} decimal FF5 daily rows from {default_cfg.cache_path}.")
