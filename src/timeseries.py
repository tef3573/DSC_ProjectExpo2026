"""
src/timeseries.py
Parametric Time-Series Volatility & Regime Modeling for Black CEO Portfolio Backtest.

Sub-issue #250:
- GARCH(1,1) Maximum Likelihood conditional volatility estimator & persistence analytics
- GJR-GARCH(1,1) asymmetric leverage volatility estimator (testing downside shock asymmetry)
- Pre-2010 vs. Post-2010 sub-period GARCH & return/risk attribution (CAGR, Sharpe, Sortino, MDD)
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE_ANNUAL: float = 0.015
MAX_PERSISTENCE: float = 0.9990
MIN_VARIANCE: float = 1e-12


@dataclass(frozen=True)
class GarchCoefficients:
    """Core fitted mean and variance equation coefficients."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float


@dataclass(frozen=True)
class GarchParams:
    """Estimated conditional variance coefficients and derived persistence diagnostics."""

    coeffs: GarchCoefficients
    persistence: float
    half_life_days: float
    unconditional_ann_vol: float

    @property
    def mu(self) -> float:
        """Mean return intercept."""
        return self.coeffs.mu

    @property
    def omega(self) -> float:
        """Baseline variance intercept."""
        return self.coeffs.omega

    @property
    def alpha(self) -> float:
        """Symmetric ARCH shock reaction coefficient."""
        return self.coeffs.alpha

    @property
    def gamma(self) -> float:
        """Asymmetric GJR downside shock leverage coefficient."""
        return self.coeffs.gamma

    @property
    def beta(self) -> float:
        """Autoregressive GARCH variance persistence coefficient."""
        return self.coeffs.beta


@dataclass(frozen=True)
class GarchFitResult:
    """Immutable summary of a fitted conditional volatility model."""

    series_name: str
    n_obs: int
    log_likelihood: float
    params: GarchParams
    conditional_vol_daily: pd.Series
    conditional_vol_annualized: pd.Series

    def to_dict(self) -> Dict[str, float]:
        """Serialize scalar parameter estimates and diagnostics to a JSON-safe dict."""
        return {
            "n_obs": int(self.n_obs),
            "mu": float(self.params.mu),
            "omega": float(self.params.omega),
            "alpha": float(self.params.alpha),
            "gamma": float(self.params.gamma),
            "beta": float(self.params.beta),
            "persistence": float(self.params.persistence),
            "half_life_days": float(self.params.half_life_days),
            "unconditional_ann_vol": float(self.params.unconditional_ann_vol),
            "log_likelihood": float(self.log_likelihood),
        }


def _filter_garch_variance(
    residuals: np.ndarray,
    coeffs: GarchCoefficients,
    initial_var: float,
) -> np.ndarray:
    """Recursively filter conditional variances sigma_t^2 for GARCH(1,1) / GJR-GARCH(1,1)."""
    n_obs = len(residuals)
    variances = np.empty(n_obs, dtype=np.float64)
    variances[0] = max(initial_var, MIN_VARIANCE)
    sq_resid = residuals * residuals
    neg_indicator = (residuals < 0.0).astype(np.float64)
    for idx in range(1, n_obs):
        shock_weight = coeffs.alpha + coeffs.gamma * neg_indicator[idx - 1]
        sigma2 = coeffs.omega + shock_weight * sq_resid[idx - 1] + coeffs.beta * variances[idx - 1]
        variances[idx] = max(sigma2, MIN_VARIANCE)
    return variances


def _garch_neg_log_likelihood(
    theta: np.ndarray,
    residuals: np.ndarray,
    sample_var: float,
) -> float:
    """Compute Gaussian negative log-likelihood for symmetric GARCH(1,1)."""
    omega, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    persistence = alpha + beta
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or persistence >= MAX_PERSISTENCE:
        return float(1e6 + 1e7 * max(0.0, persistence - MAX_PERSISTENCE + 1e-4))

    coeffs = GarchCoefficients(mu=0.0, omega=omega, alpha=alpha, gamma=0.0, beta=beta)
    variances = _filter_garch_variance(residuals, coeffs, sample_var)
    log_lik = -0.5 * np.sum(
        np.log(2.0 * math.pi) + np.log(variances) + (residuals**2) / variances
    )
    return 1e9 if not math.isfinite(float(log_lik)) else float(-log_lik)


def _gjr_neg_log_likelihood(
    theta: np.ndarray,
    residuals: np.ndarray,
    sample_var: float,
) -> float:
    """Compute Gaussian negative log-likelihood for asymmetric GJR-GARCH(1,1)."""
    omega, alpha, gamma, beta = (
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
        float(theta[3]),
    )
    persistence = alpha + 0.5 * gamma + beta
    if (
        omega <= 0.0
        or alpha < 0.0
        or alpha + gamma < 0.0
        or beta < 0.0
        or persistence >= MAX_PERSISTENCE
    ):
        return float(1e6 + 1e7 * max(0.0, persistence - MAX_PERSISTENCE + 1e-4))

    coeffs = GarchCoefficients(mu=0.0, omega=omega, alpha=alpha, gamma=gamma, beta=beta)
    variances = _filter_garch_variance(residuals, coeffs, sample_var)
    log_lik = -0.5 * np.sum(
        np.log(2.0 * math.pi) + np.log(variances) + (residuals**2) / variances
    )
    return 1e9 if not math.isfinite(float(log_lik)) else float(-log_lik)


def _optimize_garch_theta(residuals: np.ndarray, sample_var: float) -> Tuple[np.ndarray, float]:
    """Run multi-start L-BFGS-B optimization for symmetric GARCH(1,1) parameters."""
    candidates: Tuple[Tuple[float, float, float], ...] = (
        (0.05 * sample_var, 0.08, 0.87),
        (0.02 * sample_var, 0.12, 0.85),
        (0.10 * sample_var, 0.05, 0.85),
    )
    bounds = [(1e-12, 10.0 * sample_var), (1e-6, 0.50), (1e-6, 0.990)]
    best_theta = np.array(candidates[0], dtype=np.float64)
    best_nll = float("inf")
    for init_omega, init_alpha, init_beta in candidates:
        opt = minimize(
            _garch_neg_log_likelihood,
            x0=np.array([init_omega, init_alpha, init_beta], dtype=np.float64),
            args=(residuals, sample_var),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if float(opt.fun) < best_nll:
            best_nll = float(opt.fun)
            best_theta = np.asarray(opt.x, dtype=np.float64)
    return best_theta, -best_nll


def _optimize_gjr_theta(residuals: np.ndarray, sample_var: float) -> Tuple[np.ndarray, float]:
    """Run multi-start L-BFGS-B optimization for asymmetric GJR-GARCH(1,1) parameters."""
    candidates: Tuple[Tuple[float, float, float, float], ...] = (
        (0.05 * sample_var, 0.04, 0.10, 0.85),
        (0.02 * sample_var, 0.06, 0.14, 0.82),
        (0.08 * sample_var, 0.02, 0.08, 0.88),
    )
    bounds = [
        (1e-12, 10.0 * sample_var),
        (1e-6, 0.45),
        (-0.20, 0.60),
        (1e-6, 0.990),
    ]
    best_theta = np.array(candidates[0], dtype=np.float64)
    best_nll = float("inf")
    for init_vals in candidates:
        opt = minimize(
            _gjr_neg_log_likelihood,
            x0=np.array(init_vals, dtype=np.float64),
            args=(residuals, sample_var),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-10},
        )
        if float(opt.fun) < best_nll:
            best_nll = float(opt.fun)
            best_theta = np.asarray(opt.x, dtype=np.float64)
    return best_theta, -best_nll


def _build_garch_params(raw: GarchCoefficients) -> GarchParams:
    """Project parameters onto covariance-stationary bounds and compute diagnostics."""
    omega_hat = float(max(raw.omega, MIN_VARIANCE))
    alpha_hat = float(max(raw.alpha, 0.0))
    gamma_hat = float(max(raw.gamma, -alpha_hat))
    beta_hat = float(max(raw.beta, 0.0))

    raw_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat
    if raw_persistence >= MAX_PERSISTENCE:
        scale = (MAX_PERSISTENCE - 1e-4) / raw_persistence
        alpha_hat *= scale
        gamma_hat *= scale
        beta_hat *= scale

    persistence = float(alpha_hat + 0.5 * gamma_hat + beta_hat)
    half_life = (
        float(math.log(0.5) / math.log(persistence))
        if 0.0 < persistence < 1.0
        else float("inf")
    )
    uncond_ann_vol = float(
        math.sqrt((omega_hat / max(1.0 - persistence, 1e-6)) * TRADING_DAYS_PER_YEAR)
    )
    return GarchParams(
        coeffs=GarchCoefficients(
            mu=raw.mu,
            omega=omega_hat,
            alpha=alpha_hat,
            gamma=gamma_hat,
            beta=beta_hat,
        ),
        persistence=persistence,
        half_life_days=half_life,
        unconditional_ann_vol=uncond_ann_vol,
    )


def _prepare_clean_returns(
    returns: pd.Series,
    active_mask: Optional[pd.Series] = None,
) -> pd.Series:
    """Drop missing observations and optionally restrict to active market-invested sessions."""
    clean = returns.dropna().astype(np.float64)
    if active_mask is not None:
        aligned_mask = active_mask.reindex(clean.index).fillna(False).astype(bool)
        clean = clean.loc[aligned_mask]
    if len(clean) < 30:
        raise ValueError(
            f"GARCH estimation requires at least 30 observations; got {len(clean)}."
        )
    return clean


def fit_garch_11(
    returns: pd.Series,
    series_name: Optional[str] = None,
    active_mask: Optional[pd.Series] = None,
) -> GarchFitResult:
    """Estimate a stationary symmetric GARCH(1,1) model via Gaussian MLE (#250)."""
    clean = _prepare_clean_returns(returns, active_mask=active_mask)
    values = clean.to_numpy(dtype=np.float64)
    mu_hat = float(np.mean(values))
    residuals = values - mu_hat
    sample_var = float(max(np.var(residuals, ddof=1), MIN_VARIANCE))

    theta, log_lik = _optimize_garch_theta(residuals, sample_var)
    params = _build_garch_params(
        GarchCoefficients(
            mu=mu_hat,
            omega=float(theta[0]),
            alpha=float(theta[1]),
            gamma=0.0,
            beta=float(theta[2]),
        )
    )

    daily_std = np.sqrt(_filter_garch_variance(residuals, params.coeffs, sample_var))
    label = series_name or str(clean.name or "Return_Series")
    return GarchFitResult(
        series_name=label,
        n_obs=len(clean),
        log_likelihood=log_lik,
        params=params,
        conditional_vol_daily=pd.Series(daily_std, index=clean.index, name=f"{label}_Vol_Daily"),
        conditional_vol_annualized=pd.Series(
            daily_std * math.sqrt(TRADING_DAYS_PER_YEAR),
            index=clean.index,
            name=f"{label}_Vol_Ann",
        ),
    )


def fit_gjr_garch_11(
    returns: pd.Series,
    series_name: Optional[str] = None,
    active_mask: Optional[pd.Series] = None,
) -> GarchFitResult:
    """Estimate an asymmetric GJR-GARCH(1,1) leverage model via Gaussian MLE (#250)."""
    clean = _prepare_clean_returns(returns, active_mask=active_mask)
    values = clean.to_numpy(dtype=np.float64)
    mu_hat = float(np.mean(values))
    residuals = values - mu_hat
    sample_var = float(max(np.var(residuals, ddof=1), MIN_VARIANCE))

    theta, log_lik = _optimize_gjr_theta(residuals, sample_var)
    params = _build_garch_params(
        GarchCoefficients(
            mu=mu_hat,
            omega=float(theta[0]),
            alpha=float(theta[1]),
            gamma=float(theta[2]),
            beta=float(theta[3]),
        )
    )

    daily_std = np.sqrt(_filter_garch_variance(residuals, params.coeffs, sample_var))
    label = series_name or str(clean.name or "Return_Series")
    return GarchFitResult(
        series_name=label,
        n_obs=len(clean),
        log_likelihood=log_lik,
        params=params,
        conditional_vol_daily=pd.Series(daily_std, index=clean.index, name=f"{label}_GJR_Daily"),
        conditional_vol_annualized=pd.Series(
            daily_std * math.sqrt(TRADING_DAYS_PER_YEAR),
            index=clean.index,
            name=f"{label}_GJR_Ann",
        ),
    )


def compute_subperiod_performance_metrics(
    returns: pd.Series,
    risk_free_annual: float = RISK_FREE_RATE_ANNUAL,
) -> Dict[str, float]:
    """Compute CAGR, Volatility, Sharpe, Sortino, and Max Drawdown on a return slice."""
    clean = returns.dropna().astype(np.float64)
    n_obs = len(clean)
    if n_obs == 0:
        return {
            "n_obs": 0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    cum_curve = (1.0 + clean).cumprod()
    years = n_obs / float(TRADING_DAYS_PER_YEAR)
    cagr = float(float(cum_curve.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0.0 else 0.0

    daily_std = float(clean.std(ddof=1))
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = clean - rf_daily
    downside_dev = float(
        np.sqrt(np.mean(np.minimum(excess.to_numpy(dtype=np.float64), 0.0) ** 2))
    )
    ann_factor = math.sqrt(TRADING_DAYS_PER_YEAR)
    mdd = float(abs(((cum_curve - cum_curve.cummax()) / cum_curve.cummax()).min()))

    return {
        "n_obs": int(n_obs),
        "cagr": cagr,
        "annualized_volatility": float(daily_std * ann_factor),
        "sharpe_ratio": float((excess.mean() / daily_std) * ann_factor) if daily_std > 0.0 else 0.0,
        "sortino_ratio": (
            float((excess.mean() / downside_dev) * ann_factor) if downside_dev > 0.0 else 0.0
        ),
        "max_drawdown": mdd,
    }


def fit_portfolio_garch_suite(
    results_path: Path = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv",
) -> Tuple[Dict[str, GarchFitResult], Dict[str, GarchFitResult]]:
    """Fit symmetric GARCH(1,1) and asymmetric GJR-GARCH(1,1) suites across strategies & SPY."""
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    active_sessions = (
        df["Active_Constituents_Nt"] > 0 if "Active_Constituents_Nt" in df.columns else None
    )
    target_cols = {
        "Black_CEO_Net": ("Black CEO Equal-Weight (Net)", active_sessions),
        "Convex_Quality_Net": ("Black CEO Convex-Quality (Net)", active_sessions),
        "SPY_Daily": ("SPY (S&P 500 Cap-Weighted)", None),
    }
    garch_models: Dict[str, GarchFitResult] = {}
    gjr_models: Dict[str, GarchFitResult] = {}
    for col, (label, mask) in target_cols.items():
        if col in df.columns:
            garch_models[col] = fit_garch_11(df[col], series_name=label, active_mask=mask)
            gjr_models[col] = fit_gjr_garch_11(df[col], series_name=label, active_mask=mask)
    return garch_models, gjr_models


def _evaluate_single_regime(
    sub_df: pd.DataFrame,
    regime_key: str,
) -> Tuple[Dict[str, GarchFitResult], Dict[str, Dict[str, float]]]:
    """Fit GARCH(1,1) models and compute return metrics for a single regime slice."""
    active_mask = (
        sub_df["Active_Constituents_Nt"] > 0
        if "Active_Constituents_Nt" in sub_df.columns
        else None
    )
    target_cols = {
        "Black_CEO_Net": "Black CEO Equal-Weight (Net)",
        "Convex_Quality_Net": "Black CEO Convex-Quality (Net)",
        "SPY_Daily": "SPY (S&P 500 Cap-Weighted)",
    }
    regime_models: Dict[str, GarchFitResult] = {}
    regime_perf: Dict[str, Dict[str, float]] = {}
    for col, label in target_cols.items():
        if col in sub_df.columns:
            mask = None if col == "SPY_Daily" else active_mask
            regime_models[col] = fit_garch_11(
                sub_df[col], series_name=f"{label} [{regime_key}]", active_mask=mask
            )
            regime_perf[col] = compute_subperiod_performance_metrics(sub_df[col])
    return regime_models, regime_perf


def fit_subperiod_garch_attribution(
    results_path: Path = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv",
    split_date: str = "2010-01-01",
) -> Tuple[Dict[str, Dict[str, GarchFitResult]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Split backtest into Pre-2010 and Post-2010 regimes for GARCH and performance attribution."""
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    cutoff = pd.Timestamp(split_date)
    slices = {
        "pre_2010": df.loc[df.index < cutoff],
        "post_2010": df.loc[df.index >= cutoff],
    }
    garch_attr: Dict[str, Dict[str, GarchFitResult]] = {}
    perf_attr: Dict[str, Dict[str, Dict[str, float]]] = {}
    for regime_key, sub_df in slices.items():
        garch_attr[regime_key], perf_attr[regime_key] = _evaluate_single_regime(sub_df, regime_key)
    return garch_attr, perf_attr


def export_garch_to_summary_metrics(
    garch_suite: Dict[str, GarchFitResult],
    gjr_suite: Optional[Dict[str, GarchFitResult]] = None,
    subperiod_garch: Optional[Dict[str, Dict[str, GarchFitResult]]] = None,
    subperiod_perf: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    summary_path: Path = Path(__file__).resolve().parent.parent / "data" / "summary_metrics.json",
) -> Path:
    """Persist GARCH(1,1), GJR-GARCH(1,1), and sub-period attribution to summary_metrics.json."""
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["timeseries_garch"] = {key: res.to_dict() for key, res in garch_suite.items()}
    if gjr_suite is not None:
        payload["timeseries_gjr_garch"] = {key: res.to_dict() for key, res in gjr_suite.items()}
    if subperiod_garch is not None:
        payload["timeseries_subperiods"] = {
            regime: {key: res.to_dict() for key, res in models.items()}
            for regime, models in subperiod_garch.items()
        }
    if subperiod_perf is not None:
        payload["subperiod_performance"] = subperiod_perf
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary_path


def _cli_main() -> None:
    """Run GARCH(1,1), GJR-GARCH(1,1), and sub-period attribution, then print tables."""
    from src.backtest import format_scorecard_table  # pylint: disable=import-outside-toplevel

    garch_suite, gjr_suite = fit_portfolio_garch_suite()
    subperiod_garch, subperiod_perf = fit_subperiod_garch_attribution()
    out_path = export_garch_to_summary_metrics(
        garch_suite,
        gjr_suite=gjr_suite,
        subperiod_garch=subperiod_garch,
        subperiod_perf=subperiod_perf,
    )

    print("--- Full-Horizon Symmetric GARCH(1,1) MLE (2000–2026) ---")
    for key, res in garch_suite.items():
        par = res.params
        print(
            f"{key:20s} | alpha={par.alpha:.4f}, beta={par.beta:.4f}, "
            f"a+b={par.persistence:.4f}, half-life={par.half_life_days:.1f}d, "
            f"uncond_vol={par.unconditional_ann_vol * 100:.2f}%"
        )

    print("\n--- Full-Horizon Asymmetric GJR-GARCH(1,1) Leverage MLE (2000–2026) ---")
    for key, res in gjr_suite.items():
        par = res.params
        print(
            f"{key:20s} | alpha={par.alpha:.4f}, gamma={par.gamma:.4f}, beta={par.beta:.4f}, "
            f"eff_pers={par.persistence:.4f}, half-life={par.half_life_days:.1f}d, "
            f"uncond_vol={par.unconditional_ann_vol * 100:.2f}%"
        )

    print("\n--- Sub-Period Performance & GARCH Attribution (Pre-2010 vs. Post-2010) ---")
    for regime, models in subperiod_garch.items():
        for key, res in models.items():
            par = res.params
            prf = subperiod_perf[regime][key]
            print(
                f"{regime:9s} | {key:18s} | CAGR={prf['cagr']:6.2%}, "
                f"Sharpe={prf['sharpe_ratio']:4.2f}, Sortino={prf['sortino_ratio']:4.2f}, "
                f"MDD={prf['max_drawdown']:6.2%} | a+b={par.persistence:.4f}, "
                f"HL={par.half_life_days:5.1f}d"
            )

    print("\n" + format_scorecard_table(json.loads(out_path.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    _cli_main()
