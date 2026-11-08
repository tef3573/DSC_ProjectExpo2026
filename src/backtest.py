"""
src/backtest.py
Vectorized Portfolio Backtesting Engine with Institutional Friction and Artifact Export.
- Sub-issues #104, #106, #107, #108, #110: Core simulation and friction mechanics.
- Sub-issue #127: Export comprehensive daily time series to data/backtest_results.csv.
"""
from pathlib import Path
from typing import Tuple, Optional
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_EXPORT_PATH = DATA_DIR / "backtest_results.csv"


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes arithmetic daily percentage returns."""
    return prices.pct_change().fillna(0.0)


def compute_equal_weights(
    universe_mask: pd.DataFrame,
    min_diversification: int = 3
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Computes dynamic equal target weights with cash-drag concentration defense."""
    n_t = universe_mask.sum(axis=1)
    stock_weights = pd.DataFrame(0.0, index=universe_mask.index, columns=universe_mask.columns)

    normal_mask = n_t >= min_diversification
    for col in universe_mask.columns:
        stock_weights.loc[normal_mask, col] = np.where(
            universe_mask.loc[normal_mask, col],
            1.0 / n_t.loc[normal_mask],
            0.0
        )

    defensive_mask = (n_t > 0) & (n_t < min_diversification)
    for col in universe_mask.columns:
        stock_weights.loc[defensive_mask, col] = np.where(
            universe_mask.loc[defensive_mask, col],
            1.0 / min_diversification,
            0.0
        )

    equity_total = stock_weights.sum(axis=1)
    cash_weights = 1.0 - equity_total
    return stock_weights, cash_weights, n_t


def compute_weight_drift(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    portfolio_gross_returns: pd.Series
) -> pd.DataFrame:
    """Computes post-return drifted weights at the close of trading session t."""
    aligned_returns = asset_returns[target_weights.columns]
    num = target_weights * (1.0 + aligned_returns)
    denom = 1.0 + portfolio_gross_returns.values[:, np.newaxis]
    denom = np.where(denom == 0.0, 1.0, denom)
    return num / denom


def compute_turnover_and_costs(
    target_weights: pd.DataFrame,
    drifted_weights: pd.DataFrame,
    fee_bps: float = 10.0
) -> Tuple[pd.Series, pd.Series]:
    """Computes one-way portfolio rebalance turnover and fee deductions."""
    cost_rate = fee_bps / 10000.0
    drifted_prior = drifted_weights.shift(1).fillna(0.0)

    equity_diff = (target_weights - drifted_prior).abs().sum(axis=1)
    target_cash = 1.0 - target_weights.sum(axis=1)
    drifted_cash = 1.0 - drifted_prior.sum(axis=1)
    cash_diff = (target_cash - drifted_cash).abs()

    one_way_turnover = 0.5 * (equity_diff + cash_diff)
    one_way_turnover.iloc[0] = 0.0

    costs = one_way_turnover * cost_rate
    costs.iloc[0] = 0.0
    return one_way_turnover, costs


def export_backtest_results(
    results_df: pd.DataFrame,
    export_path: Path = DEFAULT_EXPORT_PATH
) -> Path:
    """
    Sub-issue #127: Formats and exports complete daily backtest time series.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(export_path, index=True)
    logging.info(f"Sub-issue #127: Exported daily time series to {export_path}")
    return export_path


def run_backtest(
    prices: pd.DataFrame,
    mask: pd.DataFrame,
    fee_bps: float = 10.0,
    rf_annual: float = 0.015,
    min_diversification: int = 3,
    export_results: bool = True
) -> pd.DataFrame:
    """Executes full vectorized point-in-time backtest and generates daily performance curves."""
    returns = compute_returns(prices)
    common_idx = prices.index.intersection(mask.index)
    prices = prices.loc[common_idx]
    mask = mask.loc[common_idx]
    returns = returns.loc[common_idx]

    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    stock_weights, cash_weights, n_t = compute_equal_weights(mask, min_diversification=min_diversification)

    portfolio_gross = pd.Series(0.0, index=common_idx)
    drifted_weights = pd.DataFrame(0.0, index=common_idx, columns=stock_weights.columns)

    for i, date in enumerate(common_idx):
        day_w = stock_weights.loc[date]
        day_r = returns.loc[date, stock_weights.columns]
        cash_w = cash_weights.loc[date]
        gross_ret = float(np.sum(day_w * day_r) + cash_w * rf_daily)
        portfolio_gross.iloc[i] = gross_ret

        denom = 1.0 + gross_ret
        drifted_w = (day_w * (1.0 + day_r)) / (denom if denom != 0.0 else 1.0)
        drifted_weights.iloc[i] = drifted_w

    turnover, costs = compute_turnover_and_costs(stock_weights, drifted_weights, fee_bps=fee_bps)
    portfolio_net = portfolio_gross - costs

    net_cum = (1.0 + portfolio_net).cumprod()
    strat_peak = net_cum.cummax()
    strat_dd = (net_cum - strat_peak) / strat_peak

    spy_ret = returns["SPY"] if "SPY" in returns.columns else pd.Series(0.0, index=common_idx)
    spy_cum = (1.0 + spy_ret).cumprod()
    spy_peak = spy_cum.cummax()
    spy_dd = (spy_cum - spy_peak) / spy_peak

    rsp_ret = returns["RSP"] if "RSP" in returns.columns else pd.Series(0.0, index=common_idx)
    rsp_cum = (1.0 + rsp_ret).cumprod()
    rsp_peak = rsp_cum.cummax()
    rsp_dd = (rsp_cum - rsp_peak) / rsp_peak

    df_out = pd.DataFrame({
        "Black_CEO_Gross": portfolio_gross,
        "Black_CEO_Net": portfolio_net,
        "Black_CEO_Cumulative": net_cum,
        "Cash_Weight": cash_weights,
        "Active_Constituents_Nt": n_t,
        "One_Way_Turnover": turnover,
        "Transaction_Costs": costs,
        "Strategy_Drawdown": strat_dd,
        "SPY_Daily": spy_ret,
        "SPY_Cumulative": spy_cum,
        "SPY_Drawdown": spy_dd,
        "RSP_Daily": rsp_ret,
        "RSP_Cumulative": rsp_cum,
        "RSP_Drawdown": rsp_dd,
    }, index=common_idx)

    if export_results:
        export_backtest_results(df_out)

    return df_out


if __name__ == "__main__":
    from src.data_loader import load_prices, load_universe_mask
    prices_df = load_prices()
    mask_df = load_universe_mask()
    run_backtest(prices_df, mask_df, export_results=True)
