"""
Point-in-Time Universe Masking Module
-------------------------------------
Constructs a dynamic T x N boolean mask matrix where an asset is marked True
only during verified tenure dates, preventing look-ahead and survivorship bias.
"""

from typing import Tuple
import pandas as pd  # pyright: ignore[reportMissingModuleSource]

ROSTER_PATH = "data/black_ceo_roster.csv"
PRICES_PATH = "data/prices_daily.csv"


def create_point_in_time_mask(
    roster_path: str = ROSTER_PATH, prices_path: str = PRICES_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads daily prices and builds the investable universe mask.

    Parameters:
        roster_path (str): Path to the verified tenure roster CSV.
        prices_path (str): Path to historical daily prices CSV.

    Returns:
        stock_prices (pd.DataFrame): Daily adjusted prices for universe stocks.
        mask (pd.DataFrame): Boolean DataFrame (True only when stock is investable).
        benchmarks (pd.DataFrame): Daily prices for SPY and RSP.
    """
    # 1. Load price matrix with parsed datetime index
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    # 2. Extract benchmarks and isolate asset prices
    available_bms = [c for c in ["SPY", "RSP"] if c in prices.columns]
    benchmarks = prices[available_bms].copy()
    stock_prices = prices.drop(columns=["SPY", "RSP"], errors="ignore").sort_index()

    # 3. Load executive tenure roster
    roster = pd.read_csv(roster_path, parse_dates=["start_date", "end_date"])

    # 4. Initialize empty boolean mask matching asset prices dimensions
    mask = pd.DataFrame(False, index=stock_prices.index, columns=stock_prices.columns)

    # 5. Populate True for active CEO tenures
    for _, row in roster.iterrows():
        ticker = row["ticker"]
        start_dt = row["start_date"]
        end_dt = row["end_date"]

        if ticker in mask.columns:
            # Set True for dates strictly within [start_date, end_date]
            mask.loc[(mask.index >= start_dt) & (mask.index <= end_dt), ticker] = True

    return stock_prices, mask, benchmarks


if __name__ == "__main__":
    stock_prices, mask, benchmarks = create_point_in_time_mask()
    active_counts = mask.sum(axis=1)

    print("-" * 60)
    print(" POINT-IN-TIME UNIVERSE MASK SUMMARY")
    print("-" * 60)
    print(f"Date Range:              {mask.index.min().date()} to {mask.index.max().date()}")
    print(f"Total Trading Days:      {len(mask)}")
    print(f"Total Unique Assets:     {mask.shape[1]}")
    print(f"Max Concurrent CEOs:     {active_counts.max()}")
    print(f"Min Concurrent CEOs:     {active_counts.min()}")
    print(f"Average Active CEOs:     {active_counts.mean():.2f}")
    print(f"Benchmarks Available:    {list(benchmarks.columns)}")
    print("-" * 60)