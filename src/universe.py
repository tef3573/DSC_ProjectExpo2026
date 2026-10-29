"""
src/universe.py
Point-in-time eligibility mask matrix builder.
"""
from typing import Tuple
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ROSTER_PATH = BASE_DIR / "data" / "black_ceo_roster.csv"
PRICES_PATH = BASE_DIR / "data" / "prices_daily.csv"


def create_point_in_time_mask(
    roster_path: str = str(ROSTER_PATH), 
    prices_path: str = str(PRICES_PATH)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constructs the T x N eligibility mask.
    Applies t+1 trading day execution lag to eliminate look-ahead bias.
    """
    roster = pd.read_csv(roster_path)
    roster.columns = roster.columns.str.strip()
    for col in roster.select_dtypes(include="object").columns:
        roster[col] = roster[col].str.strip()

    roster["start_date"] = pd.to_datetime(roster["start_date"])
    roster["end_date"] = pd.to_datetime(roster["end_date"].fillna("2026-10-27"))

    # If prices_daily.csv exists, load and align; otherwise build schema
    if Path(prices_path).exists():
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        available_bms = [c for c in ["SPY", "RSP"] if c in prices.columns]
        benchmarks = prices[available_bms].copy()
        stock_prices = prices.drop(columns=["SPY", "RSP"], errors="ignore").sort_index()
    else:
        # Fallback for fresh CI environments before price ingestion
        dates = pd.date_range("2000-01-01", "2026-10-27", freq="B")
        tickers = sorted(roster["ticker"].unique().tolist())
        stock_prices = pd.DataFrame(index=dates, columns=tickers)
        benchmarks = pd.DataFrame(index=dates, columns=["SPY", "RSP"])

    mask = pd.DataFrame(False, index=stock_prices.index, columns=stock_prices.columns)

    for _, row in roster.iterrows():
        ticker = row["ticker"]
        start_dt = row["start_date"]
        end_dt = row["end_date"]

        if ticker in mask.columns:
            # Shift entry date to t+1 to eliminate same-day look-ahead bias
            active_dates = (mask.index > start_dt) & (mask.index <= end_dt)
            mask.loc[active_dates, ticker] = mask.loc[active_dates, ticker] | True

    return stock_prices, mask, benchmarks


if __name__ == "__main__":
    stock_prices, mask, benchmarks = create_point_in_time_mask()
    active_counts = mask.sum(axis=1)

    print("=" * 60)
    print(" POINT-IN-TIME UNIVERSE MASK SUMMARY")
    print("=" * 60)
    print(f"Date Range:          {mask.index.min().date()} to {mask.index.max().date()}")
    print(f"Total Trading Days:  {len(mask)}")
    print(f"Total Constituents:  {mask.shape[1]}")
    print(f"Max Active CEOs:     {active_counts.max()}")
    print(f"Min Active CEOs:     {active_counts.min()}")
    print(f"Mean Active CEOs:    {active_counts.mean():.2f}")
    print(f"Benchmarks:          {list(benchmarks.columns)}")
    print("=" * 60)
