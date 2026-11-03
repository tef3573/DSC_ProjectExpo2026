"""
src/backtest.py
Base Vectorized Portfolio Backtesting Engine.
- Issue #87 / Sub-issue #94: Daily 1/Nt asset return dot product.
- Issue #87 / Sub-issue #95: Dynamic cash drag allocation when Nt == 0.
- Issue #87 / Sub-issue #96: Division-by-zero protections during holidays and inactive periods.
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

# Loads clean price matrix and lagged universe mask.
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices daily matrix missing at {PRICES_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Universe mask missing at {MASK_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True)
    return prices, mask

"""
    Computes daily arithmetic simple returns: r_{i,t} = (P_{i,t} / P_{i,t-1}) - 1.
    Handles exchange halts and zero-volume days cleanly without division-by-zero errors.
"""
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    # Replace inf and -inf values with NaN, then fill single-day gaps with 0.0
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return returns

"""
Sub-issue #94 - 96: Vectorized daily portfolio 
return computation with dynamic cash drag allocation.
"""
def compute_equal_weights(mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Sub-issue #94: Compute daily 1/Nt weights for active constituents
    n_t = mask.sum(axis=1)

    # Sub-issue #95: Dynamic cash drag allocation when Nt == 0, Ensure numerical floating precision clamp [0.0, 1.0]
    stock_weights = mask.div(n_t, axis=0).fillna(0.0).clip(0.0, 1.0)

    # Sub-issue #96: Division-by-zero protections during holidays and inactive periods
    cash_weights = 1.0 - stock_weights.sum(axis=1).clip(0.0, 1.0)
    return stock_weights, cash_weights, n_t


# Sub-issue #94 - 96: Run Vectorized daily portfolio with dynamic cash drag allocation.
def run_backtest(
    prices: pd.DataFrame,
    mask: pd.DataFrame,
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    all_returns = compute_returns(prices)
    constituent_cols = [c for c in mask.columns if c in all_returns.columns]

    stock_returns = all_returns[constituent_cols]
    stock_weights, cash_weights, n_t = compute_equal_weights(mask[constituent_cols])

    # Sub-issue #94: Matrix dot product across active constituents
    portfolio_stock_returns = (stock_weights * stock_returns).sum(axis=1)

    # Sub-issue #95: Incorporate cash returns during zero active periods
    portfolio_cash_returns = cash_weights * rf_daily
    total_portfolio_returns = portfolio_stock_returns + portfolio_cash_returns

    perf = pd.DataFrame(index=prices.index)
    perf["Black_CEO_EW"] = total_portfolio_returns
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
        f"Cumulative Multiples: Strategy={perf['Black_CEO_Cumulative'].iloc[-1]:.3f}x | "
        f"SPY={perf['SPY_Cumulative'].iloc[-1]:.3f}x | RSP={perf['RSP_Cumulative'].iloc[-1]:.3f}x"
    )
    return perf

# Exports daily returns and wealth series to CSV.
def export_results(perf: pd.DataFrame, path: Path = OUTPUT_RETURNS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(path, float_format="%.6f")
    logging.info(f"Saved portfolio daily returns to: {path}")
    return path

# Print
if __name__ == "__main__":
    df_prices, df_mask = load_inputs()
    perf_results = run_backtest(df_prices, df_mask, rf_daily=0.0)
    export_results(perf_results)

    print("\n--- ISSUE #87 ENGINE VALIDATION ---")
    print(f"Total Sessions: {len(perf_results)}")
    print(f"Zero Active Sessions (Cash Held): {(perf_results['Active_Constituents'] == 0).sum()}")
    print("Final Growth Multiples:")
    print(f"  Black CEO Strategy: {perf_results['Black_CEO_Cumulative'].iloc[-1]:.3f}x")
    print(f"  SPY Benchmark:      {perf_results['SPY_Cumulative'].iloc[-1]:.3f}x")
    print(f"  RSP Benchmark:      {perf_results['RSP_Cumulative'].iloc[-1]:.3f}x")