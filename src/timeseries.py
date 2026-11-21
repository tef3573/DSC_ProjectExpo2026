"""
src/timeseries.py
Parametric Time-Series Volatility & Regime Modeling for Black CEO Portfolio Backtest.

Sub-issue #250: GARCH(1,1) Maximum Likelihood conditional volatility estimator,
persistence diagnostics, and shock half-life analytics.
"""
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR: int = 252
MAX_PERSISTENCE: float = 0.9995
MIN_VARIANCE: float = 1e-12


@dataclass(frozen=True)
class GarchFitResult:
    """Immutable summary of a fitted GARCH(1,1) conditional volatility model."""

    series_name: str
    n_obs: int
    mu: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    half_life_days: float
    unconditional_ann_vol: float
    log_likelihood: float
    conditional_vol_daily: pd.Series
    conditional_vol_annualized: pd.Series

    def to_dict(self) -> Dict[str, float]:
        """Serialize scalar parameter estimates and diagnostics to a JSON-safe dict."""
        return {
            "n_obs": int(self.n_obs),
            "mu": float(self.mu),
            "omega": float(self.omega),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "persistence": float(self.persistence),
            "half_life_days": float(self.half_life_days),
            "unconditional_ann_vol": float(self.unconditional_ann_vol),
            "log_likelihood": float(self.log_likelihood),
        }


def _filter_garch_variance(
    residuals: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    initial_var: float,
) -> np.ndarray:
    """Recursively filter conditional variances sigma_t^2 for a GARCH(1,1) process."""
    n_obs = len(residuals)
    variances = np.empty(n_obs, dtype=np.float64)
    variances[0] = max(initial_var, MIN_VARIANCE)
    sq_resid = residuals * residuals
    for idx in range(1, n_obs):
        sigma2 = omega + alpha * sq_resid[idx - 1] + beta * variances[idx - 1]
        variances[idx] = max(sigma2, MIN_VARIANCE)
    return variances


def _garch_neg_log_likelihood(
    params: np.ndarray,
    residuals: np.ndarray,
    sample_var: float,
) -> float:
    """Compute Gaussian negative log-likelihood with smooth stationarity barrier."""
    omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    persistence = alpha + beta
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or persistence >= MAX_PERSISTENCE:
        penalty = 1e6 + 1e7 * max(0.0, persistence - MAX_PERSISTENCE + 1e-4)
        return float(penalty)

    variances = _filter_garch_variance(residuals, omega, alpha, beta, sample_var)
    log_lik = -0.5 * np.sum(np.log(2.0 * math.pi) + np.log(variances) + (residuals**2) / variances)
    if not math.isfinite(float(log_lik)):
        return 1e9
    return float(-log_lik)


def fit_garch_11(
    returns: pd.Series,
    series_name: Optional[str] = None,
) -> GarchFitResult:
    """
    Estimate a stationary GARCH(1,1) model via Gaussian Maximum Likelihood (#250).

    Model specification:
        r_t = mu + eps_t,   eps_t ~ N(0, sigma_t^2)
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Args:
        returns: Daily decimal return series (DatetimeIndex or RangeIndex).
        series_name: Optional human-readable identifier for reporting.

    Returns:
        GarchFitResult containing MLE parameters, persistence, half-life, and
        filtered daily/annualized conditional volatility series.
    """
    clean = returns.dropna().astype(np.float64)
    if len(clean) < 30:
        raise ValueError(f"GARCH(1,1) estimation requires at least 30 observations; got {len(clean)}.")

    label = series_name or str(clean.name or "Return_Series")
    values = clean.to_numpy(dtype=np.float64)
    mu_hat = float(np.mean(values))
    residuals = values - mu_hat
    sample_var = float(max(np.var(residuals, ddof=1), MIN_VARIANCE))

    initial_candidates: Tuple[Tuple[float, float, float], ...] = (
        (0.05 * sample_var, 0.08, 0.87),
        (0.02 * sample_var, 0.12, 0.85),
        (0.10 * sample_var, 0.05, 0.85),
    )
    bounds = [
        (1e-12, 10.0 * sample_var),
        (1e-6, 0.50),
        (1e-6, 0.995),
    ]

    best_opt = None
    for init_omega, init_alpha, init_beta in initial_candidates:
        x0 = np.array([init_omega, init_alpha, init_beta], dtype=np.float64)
        opt = minimize(
            _garch_neg_log_likelihood,
            x0=x0,
            args=(residuals, sample_var),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if best_opt is None or opt.fun < best_opt.fun:
            best_opt = opt

    assert best_opt is not None
    omega_hat = float(max(best_opt.x[0], MIN_VARIANCE))
    alpha_hat = float(max(best_opt.x[1], 0.0))
    beta_hat = float(max(best_opt.x[2], 0.0))

    # Enforce strict covariance stationarity (alpha + beta < MAX_PERSISTENCE)
    if alpha_hat + beta_hat >= MAX_PERSISTENCE:
        scale = (MAX_PERSISTENCE - 1e-4) / (alpha_hat + beta_hat)
        alpha_hat *= scale
        beta_hat *= scale

    persistence = float(alpha_hat + beta_hat)
    half_life = (
        float(math.log(0.5) / math.log(persistence))
        if 0.0 < persistence < 1.0
        else float("inf")
    )
    uncond_var = omega_hat / max(1.0 - persistence, 1e-6)
    uncond_ann_vol = float(math.sqrt(uncond_var * TRADING_DAYS_PER_YEAR))

    filtered_var = _filter_garch_variance(residuals, omega_hat, alpha_hat, beta_hat, sample_var)
    daily_vol = pd.Series(np.sqrt(filtered_var), index=clean.index, name=f"{label}_GARCH_Vol_Daily")
    ann_vol = pd.Series(
        daily_vol.to_numpy() * math.sqrt(TRADING_DAYS_PER_YEAR),
        index=clean.index,
        name=f"{label}_GARCH_Vol_Ann",
    )

    return GarchFitResult(
        series_name=label,
        n_obs=len(clean),
        mu=mu_hat,
        omega=omega_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        persistence=persistence,
        half_life_days=half_life,
        unconditional_ann_vol=uncond_ann_vol,
        log_likelihood=float(-best_opt.fun),
        conditional_vol_daily=daily_vol,
        conditional_vol_annualized=ann_vol,
    )


def fit_portfolio_garch_suite(
    results_path: Path = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv",
) -> Dict[str, GarchFitResult]:
    """Fit GARCH(1,1) models across core Black CEO strategies and SPY benchmark."""
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    target_cols = {
        "Black_CEO_Net": "Black CEO Equal-Weight (Net)",
        "Convex_Quality_Net": "Black CEO Convex-Quality (Net)",
        "SPY_Daily": "SPY (S&P 500 Cap-Weighted)",
    }
    models: Dict[str, GarchFitResult] = {}
    for col, label in target_cols.items():
        if col in df.columns:
            models[col] = fit_garch_11(df[col], series_name=label)
    return models


if __name__ == "__main__":
    suite = fit_portfolio_garch_suite()
    for key, res in suite.items():
        print(
            f"{key:20s} | alpha={res.alpha:.4f}, beta={res.beta:.4f}, "
            f"alpha+beta={res.persistence:.4f}, half-life={res.half_life_days:.1f}d, "
            f"uncond_vol={res.unconditional_ann_vol * 100:.2f}%"
        )
