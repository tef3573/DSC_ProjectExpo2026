"""
src/backtest.py
Vectorized Portfolio Backtesting Engine.
- Sub-issue #100: Proportional equity scaling when Nt < 3.
- Sub-issue #101: Dynamic routing of unallocated cash buffer into risk-free rate.
- Sub-issue #102: Vectorized active count tracking across timeline.
- Sub-issue #106: Inter-period weight drift calculation from asset price returns.
- Sub-issue #107: Deduct 10 bps transaction fee on one-way turnover.
- Sub-issue #108: Adjust turnover formulation for initial position inception (TO_0 = 0).
"""
from pathlib import Path
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"
OUTPUT_RETURNS_PATH = DATA_DIR / "portfolio_daily_returns.csv"


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads clean price matrix and lagged universe mask."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices daily matrix missing at {PRICES_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Universe mask missing at {MASK_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True)
    return prices, mask


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes daily simple arithmetic returns with numerical safety checks."""
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return returns


def compute_equal_weights(mask: pd.DataFrame, min_diversification: int = 3) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Computes daily constituent weights with cash-drag concentration defense."""
    n_t = mask.sum(axis=1)
    weights = pd.DataFrame(0.0, index=mask.index, columns=mask.columns)

    normal_mask = n_t >= min_diversification
    if normal_mask.any():
        weights.loc[normal_mask] = mask.loc[normal_mask].div(n_t.loc[normal_mask], axis=0)

    defensive_mask = (n_t > 0) & (n_t < min_diversification)
    if defensive_mask.any():
        weights.loc[defensive_mask] = mask.loc[defensive_mask].astype(float) * (1.0 / float(min_diversification))

    stock_alloc = weights.sum(axis=1)
    cash_weights = (1.0 - stock_alloc).clip(lower=0.0, upper=1.0)

    return weights, cash_weights, n_t


def compute_weight_drift(
    weights: pd.DataFrame, 
    asset_returns: pd.DataFrame, 
    portfolio_returns: pd.Series
) -> pd.DataFrame:
    """
    Sub-issue #106: Calculate inter-period weight drift from asset price returns.
    Formula: w_{i, t}^drift = w_{i, t} * (1 + r_{i, t}) / (1 + R_{p, t})
    """
    growth_per_asset = weights * (1.0 + asset_returns)
    growth_portfolio = 1.0 + portfolio_returns
    safe_growth = growth_portfolio.replace(0.0, np.nan)
    drifted_weights = growth_per_asset.div(safe_growth, axis=0).fillna(0.0)
    return drifted_weights


def compute_turnover_and_costs(
    target_weights: pd.DataFrame,
    drifted_weights: pd.DataFrame,
    fee_bps: float = 10.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Sub-issues #107 & #108: Compute daily one-way portfolio turnover and transaction cost fee.
    Fix #108: Inception turnover at t=0 is forced to 0.0 to prevent artificial startup friction.
    """
    prior_drifted = drifted_weights.shift(1).fillna(0.0)

    stock_delta = (target_weights - prior_drifted).abs().sum(axis=1)

    prior_cash = 1.0 - prior_drifted.sum(axis=1)
    current_cash = 1.0 - target_weights.sum(axis=1)
    cash_delta = (current_cash - prior_cash).abs()

    # One-way turnover: 0.5 * (sum(|delta_stocks|) + |delta_cash|)
    turnover = (0.5 * (stock_delta + cash_delta)).copy()

    # Sub-issue #108: Force Day 0 to 0.0 (inception capitalization is not trading turnover)
    if len(turnover) > 0:
        turnover.iloc[0] = 0.0

    fee_decimal = fee_bps / 10000.0
    transaction_costs = turnover * fee_decimal

    return turnover, transaction_costs


def run_backtest(
    prices: pd.DataFrame, 
    mask: pd.DataFrame, 
    rf_annual: float = 0.015,
    fee_bps: float = 10.0
) -> pd.DataFrame:
    """
    Simulates portfolio returns with cash drag and 10 bps turnover friction deduction.
    """
    all_returns = compute_returns(prices)
    constituent_cols = [c for c in mask.columns if c in all_returns.columns]

    stock_returns = all_returns[constituent_cols]
    stock_weights, cash_weights, n_t = compute_equal_weights(mask[constituent_cols])

    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

    portfolio_stock_returns = (stock_weights * stock_returns).sum(axis=1)
    portfolio_cash_returns = cash_weights * rf_daily
    gross_returns = portfolio_stock_returns + portfolio_cash_returns

    drifted_weights = compute_weight_drift(stock_weights, stock_returns, gross_returns)
    turnover, transaction_costs = compute_turnover_and_costs(stock_weights, drifted_weights, fee_bps=fee_bps)

    net_returns = gross_returns - transaction_costs

    perf = pd.DataFrame(index=prices.index)
    perf["Black_CEO_Gross"] = gross_returns
    perf["Transaction_Costs"] = transaction_costs
    perf["Turnover"] = turnover
    perf["Black_CEO_Net"] = net_returns
    perf["Black_CEO_EW"] = net_returns
    perf["SPY"] = all_returns["SPY"] if "SPY" in all_returns.columns else 0.0
    perf["RSP"] = all_returns["RSP"] if "RSP" in all_returns.columns else 0.0
    perf["Active_Constituents"] = n_t
    perf["Cash_Weight"] = cash_weights

    # Cumulative wealth series (normalized to $1.00 base)
    perf["Black_CEO_Cumulative"] = (1.0 + perf["Black_CEO_EW"]).cumprod()
    perf["SPY_Cumulative"] = (1.0 + perf["SPY"]).cumprod()
    perf["RSP_Cumulative"] = (1.0 + perf["RSP"]).cumprod()

    logging.info(
        f"Backtest completed ({len(perf)} sessions). "
        f"Cumulative Multiples: Net Strategy={perf['Black_CEO_Cumulative'].iloc[-1]:.3f}x | "
        f"SPY={perf['SPY_Cumulative'].iloc[-1]:.3f}x | RSP={perf['RSP_Cumulative'].iloc[-1]:.3f}x"
    )
    return perf


def export_results(perf: pd.DataFrame, path: Path = OUTPUT_RETURNS_PATH) -> Path:
    """Exports daily returns and wealth series to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(path, float_format="%.6f")
    logging.info(f"Saved portfolio daily returns to: {path}")
    return path


if __name__ == "__main__":
    df_prices, df_mask = load_inputs()
    perf_results = run_backtest(df_prices, df_mask, rf_annual=0.015, fee_bps=10.0)
    export_results(perf_results)

    print("\n--- SUB-ISSUE #108: INCEPTION ADJUSTED TURNOVER VALIDATION ---")
    print(f"Total Sessions: {len(perf_results)}")
    print(f"Day 0 Turnover: {perf_results['Turnover'].iloc[0]:.6f}")
    print(f"Day 0 Transaction Cost: {perf_results['Transaction_Costs'].iloc[0]:.6f}")
    print(f"Mean Daily Turnover (t >= 1): {perf_results['Turnover'].iloc[1:].mean():.4%}")
    print(f"Final Wealth (Net): {perf_results['Black_CEO_Cumulative'].iloc[-1]:.3f}x")
