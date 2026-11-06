"""
src/metrics.py
Institutional Portfolio Performance Analytics Engine.

- Sub-issue #114: Correct annualization factor for Sharpe ratio numerator (252 vs sqrt(252)).
"""
import numpy as np
import pandas as pd
from typing import Tuple

"""
    Sub-issue #114: Calculates annualized Sharpe Ratio with correct scaling.

    Formula:
        mu_excess_annual = mean(r - rf_daily) * 252
        sigma_annual = std(r, ddof=1) * sqrt(252)
        Sharpe = mu_excess_annual / sigma_annual
"""
def compute_sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
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

"""
    Sub-issue #115: Computes daily downside semi-deviation divided by total sample length T.
    Formula:
        excess = r_t - rf_daily
        downside_diff = min(excess, 0)
        delta_daily = sqrt( (1 / T) * sum(downside_diff^2) )
"""
def compute_downside_deviation(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
    clean_returns = returns.dropna()
    t_total = len(clean_returns)
    if t_total < 1:
        return 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - rf_daily

    # Clip values above 0 to 0.0
    downside = np.minimum(excess_returns.values, 0.0)

    # Divide by total length T, not len(downside < 0)
    variance_downside = np.sum(downside ** 2) / t_total
    return float(np.sqrt(variance_downside))

"""
    Sub-issue #115: Annualized Sortino Ratio using total sample length T downside deviation.
    Formula:
        Sortino = (mean(excess) * 252) / (downside_deviation * sqrt(252))
"""
def compute_sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - rf_daily

    annualized_mean_excess = excess_returns.mean() * periods_per_year
    daily_downside = compute_downside_deviation(clean_returns, rf_annual=rf_annual, periods_per_year=periods_per_year)

    if daily_downside == 0.0 or np.isnan(daily_downside):
        return 0.0

    annualized_downside = daily_downside * np.sqrt(periods_per_year)
    return float(annualized_mean_excess / annualized_downside)

"""
    Sub-issue #116: Computes cumulative wealth series and running drawdown percentage series.
    
    Returns:
        wealth: Cumulative wealth curve normalized to 1.0.
        drawdown: Percentage drop from peak at each session t (DD_t <= 0).
"""
def compute_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    clean_returns = returns.dropna()
    if len(clean_returns) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    wealth = (1.0 + clean_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = (wealth - running_peak) / running_peak
    return wealth, drawdown

"""
    Sub-issue #116: Computes Maximum Drawdown (MDD) as an absolute positive decimal.
    Formula: |min(DD_t)|
"""
def compute_max_drawdown(returns: pd.Series) -> float:
    _, drawdown = compute_drawdown_series(returns)
    if len(drawdown) == 0:
        return 0.0
    min_dd = float(drawdown.min())
    return abs(min_dd)

"""
    Sub-issue #116: Computes Compound Annual Growth Rate (CAGR).
    Formula: W_T^(252 / T) - 1
"""
def compute_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean_returns = returns.dropna()
    t_total = len(clean_returns)
    if t_total == 0:
        return 0.0

    final_wealth = float((1.0 + clean_returns).prod())
    if final_wealth <= 0.0:
        return -1.0

    years = t_total / float(periods_per_year)
    cagr = (final_wealth ** (1.0 / years)) - 1.0
    return float(cagr)

"""
    Sub-issue #116: Computes Calmar Ratio (CAGR / |MDD|).
    returns 0.0 if MDD is zero to avoid division by zero.
"""
def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    mdd = compute_max_drawdown(returns)
    if mdd == 0.0:
        return 0.0

    cagr = compute_cagr(returns, periods_per_year=periods_per_year)
    return float(cagr / mdd) 