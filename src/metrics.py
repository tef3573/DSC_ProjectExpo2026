"""
src/metrics.py
Institutional Portfolio Performance Analytics Engine.

- Sub-issue #114: Correct annualization factor for Sharpe ratio numerator (252 vs sqrt(252)).
"""
import numpy as np
import pandas as pd

def compute_sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
    """
    Sub-issue #114: Calculates annualized Sharpe Ratio with correct scaling.
    
    Formula:
        mu_excess_annual = mean(r - rf_daily) * 252
        sigma_annual = std(r, ddof=1) * sqrt(252)
        Sharpe = mu_excess_annual / sigma_annual
    """
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - rf_daily

    daily_mean_excess = excess_returns.mean()
    daily_vol = clean_returns.std(ddof=1)

    if daily_vol == 0.0 or np.isnan(daily_vol):
        return 0.0

    # Numerator scales by 252, denominator scales by sqrt(252)
    annualized_mean_excess = daily_mean_excess * periods_per_year
    annualized_vol = daily_vol * np.sqrt(periods_per_year)

    return float(annualized_mean_excess / annualized_vol)