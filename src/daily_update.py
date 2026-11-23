"""
src/daily_update.py
Post-close daily pipeline runner for the Black CEO Equity Strategy (BCEOS).
Designed to execute on non-holiday NYSE weekdays after 5:30 PM ET to ingest
that day's closing prices, extend active CEO tenures, recompute backtest &
econometric diagnostics, and update plain-English terminal commentary.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
PRICES_PATH = DATA_DIR / "prices_daily.csv"

US_MARKET_HOLIDAYS = {
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
    "2027-01-01",
    "2027-01-18",
    "2027-02-15",
    "2027-03-26",
    "2027-05-31",
    "2027-06-18",
    "2027-07-05",
    "2027-09-06",
    "2027-11-25",
    "2027-12-24",
}

DELISTED_TICKERS = {"AET", "GBT", "JCP", "QRTEA", "PBPB", "WBA"}


def is_valid_post_close_window(now_et: datetime) -> tuple[bool, str]:
    """Return whether now_et is a non-holiday weekday at or after 17:30 ET."""
    date_str = now_et.strftime("%Y-%m-%d")
    if now_et.weekday() >= 5:
        return False, f"Skipping {date_str}: weekend (NYSE closed)."
    if date_str in US_MARKET_HOLIDAYS:
        return False, f"Skipping {date_str}: NYSE market holiday."
    if (now_et.hour, now_et.minute) < (17, 30):
        return False, f"Skipping {date_str} {now_et.strftime('%H:%M')} ET: before 17:30 ET close."
    return True, f"Running post-close update for {date_str} ({now_et.strftime('%H:%M')} ET)."


def extend_active_roster_dates() -> str:
    """Advance active CEO end_dates to match the latest trading session in prices_daily.csv."""
    prices_df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    latest_session = prices_df.index.max().strftime("%Y-%m-%d")

    roster_df = pd.read_csv(ROSTER_PATH)
    prior_max_end = str(roster_df["end_date"].max())
    active_mask = (roster_df["end_date"].astype(str) >= prior_max_end) & (
        ~roster_df["ticker"].astype(str).str.strip().str.upper().isin(DELISTED_TICKERS)
    )
    roster_df.loc[active_mask, "end_date"] = latest_session
    roster_df.to_csv(ROSTER_PATH, index=False)
    return latest_session


def run_step(module_name: str) -> None:
    """Execute a pipeline module using the active Python interpreter."""
    subprocess.run([sys.executable, "-m", module_name], cwd=str(BASE_DIR), check=True)


def run_daily_pipeline(force: bool = False) -> None:
    """Run full post-close market ingestion, backtest, econometrics, and UI export."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    allowed, reason = is_valid_post_close_window(now_et)
    if not allowed and not force:
        print(reason)
        return

    msg = reason if allowed else f"Force-running update at {now_et.strftime('%Y-%m-%d %H:%M')} ET."
    print(msg)
    run_step("src.data_loader")
    latest_session = extend_active_roster_dates()
    print(f"Extended active CEO roster through latest session: {latest_session}")

    for mod in (
        "src.backtest",
        "src.factors",
        "src.monte_carlo",
        "src.time_series_models",
        "src.export_web_payload",
    ):
        run_step(mod)

    print("Daily post-close update complete: dashboard_payload.json & commentary refreshed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-close 5:30 PM ET daily pipeline runner.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run immediately regardless of weekend, holiday, or pre-17:30 ET clock.",
    )
    args = parser.parse_args()
    run_daily_pipeline(force=args.force)
