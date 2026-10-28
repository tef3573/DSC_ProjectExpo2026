"""
src/universe.py
Generates the point-in-time eligibility boolean mask to prevent survivorship and look-ahead bias.
"""
from pathlib import Path
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_universe_mask(price_df: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a T x N boolean mask.
    A company enters on start_date + 1 trading day (look-ahead protection)
    and exits on end_date.
    """
    dates = price_df.index
    eligible_tickers = sorted(roster_df["ticker"].unique().tolist())
    
    # Audit for missing price series: fail loudly instead of silent skip
    missing_from_cache = [t for t in eligible_tickers if t not in price_df.columns]
    if missing_from_cache:
        raise KeyError(f"The following roster tickers are missing from price matrix: {missing_from_cache}")

    mask = pd.DataFrame(False, index=dates, columns=eligible_tickers)

    for _, row in roster_df.iterrows():
        ticker = row["ticker"]
        start = row["start_date"]
        end = row["end_date"]
        
        # Shift start by 1 trading session to avoid execution look-ahead bias
        active_window = (dates > start) & (dates <= end)
        mask.loc[active_window, ticker] = mask.loc[active_window, ticker] | True

    # Zero out dates where prices are missing or unlisted
    for t in eligible_tickers:
        mask[t] = mask[t] & price_df[t].notna()

    return mask


def get_investable_returns(prices: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily returns and applies the universe mask.
    Inactive days are NaN so they exert zero mathematical weight.
    """
    daily_returns = prices.pct_change(fill_method=None)
    common_tickers = mask.columns.intersection(daily_returns.columns)
    masked_returns = daily_returns[common_tickers].where(mask[common_tickers], np.nan)
    return masked_returns


if __name__ == "__main__":
    prices = pd.read_csv(DATA_DIR / "prices_daily.csv", index_col=0, parse_dates=True)
    roster = pd.read_csv(DATA_DIR / "black_ceo_roster.csv")
    roster.columns = roster.columns.str.strip()
    roster["start_date"] = pd.to_datetime(roster["start_date"])
    roster["end_date"] = pd.to_datetime(roster["end_date"].fillna("2026-10-27"))

    mask = build_universe_mask(prices, roster)
    print("Point-in-Time Universe Mask built successfully.")
    print(f"Mask Shape: {mask.shape}")
    print("Active constituent count over time:")
    print(mask.sum(axis=1).describe())