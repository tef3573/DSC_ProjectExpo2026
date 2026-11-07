"""
src/metrics.py
Institutional Portfolio Performance Analytics Engine.

- Sub-issue #114: Correct annualization factor for Sharpe ratio numerator (252 vs sqrt(252)).
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

"""
    Sub-issue #114: Calculates annualized Sharpe Ratio with correct scaling.

    Formula:
        mu_excess_annual = mean(r - rf_daily) * 252
        sigma_annual = std(r, ddof=1) * sqrt(252)
        Sharpe = mu_excess_annual / sigma_annual
"""

RSP_INCEPTION_DATE = "2003-05-01"

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


"""
    Sub-issue #120: Aligns strategy and benchmark series on an identical calendar window.
"""
def align_benchmark_series(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    start_date: str = None,
    end_date: str = None
) -> Tuple[pd.Series, pd.Series]:
        df = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
        if start_date is not None:
            df = df.loc[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df.loc[df.index <= pd.to_datetime(end_date)]
        return df["strategy"], df["benchmark"]

"""
    Sub-issue #121: Isolates strategy and RSP equal-weighted benchmark starting at RSP inception.
    Default inception anchor: May 1, 2003 (2003-05-01).
"""
def isolate_rsp_benchmark_series(
    strategy_returns: pd.Series,
    rsp_returns: pd.Series,
    inception_date: str = RSP_INCEPTION_DATE
) -> Tuple[pd.Series, pd.Series]:
    return align_benchmark_series(strategy_returns, rsp_returns, start_date=inception_date)











"""
    Sub-issue #120: Computes comparative institutional metrics against a reference benchmark.
    Returns: CAGR, Volatility, Sharpe, Max Drawdown, Beta, Alpha, and Tracking Error.
"""
def compute_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    strat, bench = align_benchmark_series(strategy_returns, benchmark_returns)
    if len(strat) < 2:
        return {}

    # Annualized CAGR & Vol
    strat_cagr = compute_cagr(strat, periods_per_year=periods_per_year)
    bench_cagr = compute_cagr(bench, periods_per_year=periods_per_year)
    strat_vol = float(strat.std(ddof=1) * np.sqrt(periods_per_year))
    bench_vol = float(bench.std(ddof=1) * np.sqrt(periods_per_year))

    # Sharpe ratios
    strat_sharpe = compute_sharpe_ratio(strat, rf_annual=rf_annual, periods_per_year=periods_per_year)
    bench_sharpe = compute_sharpe_ratio(bench, rf_annual=rf_annual, periods_per_year=periods_per_year)

    # Max Drawdowns
    strat_mdd = compute_max_drawdown(strat)
    bench_mdd = compute_max_drawdown(bench)
    
    # Beta and Alpha (OLS regression)
    covariance = np.cov(strat.values, bench.values)[0, 1]
    bench_variance = np.var(bench.values, ddof=1)
    beta = float(covariance / bench_variance) if bench_variance > 0 else 1.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    alpha_annual = float(((strat.mean() - rf_daily) - beta * (bench.mean() - rf_daily)) * periods_per_year)

    # Tracking error (annualized std of return spread)
    diff = strat - bench
    tracking_error = float(diff.std(ddof=1) * np.sqrt(periods_per_year))
    
    return {
        "observations": len(strat),
        "start_date": str(strat.index[0].date()) if hasattr(strat.index[0], "date") else str(strat.index[0]),
        "end_date": str(strat.index[-1].date()) if hasattr(strat.index[-1], "date") else str(strat.index[-1]),
        "strategy_cagr": strat_cagr,
        "benchmark_cagr": bench_cagr,
        "strategy_volatility": strat_vol,
        "benchmark_volatility": bench_vol,
        "strategy_sharpe": strat_sharpe,
        "benchmark_sharpe": bench_sharpe,
        "strategy_mdd": strat_mdd,
        "benchmark_mdd": bench_mdd,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "tracking_error": tracking_error
    }