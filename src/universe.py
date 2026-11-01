"""
src/universe.py
Point-in-Time Dynamic Universe Mask & Execution Lag Engine.
- Sub-issue #74: Strict t+1 execution lag enforcement.
- Sub-issue #75: Clean string type casting and roster normalization.
- Sub-issue #76: Optimized 2D boolean broadcasting across SPY trading calendar.
"""
from pathlib import Path
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
UNIVERSE_MASK_PATH = DATA_DIR / "universe_mask.csv"

# Sub-issue #75: Clean string type casting and roster normalization.
def load_clean_roster() -> pd.DataFrame:
    """Load and clean the black_ceo_roster.csv, ensuring proper date parsing and string normalization."""
    if not ROSTER_PATH.exists():
        raise FileNotFoundError(f"Roster not found at {ROSTER_PATH}")
    df = pd.read_csv(ROSTER_PATH)
    df.columns = df.columns.str.strip()
    """Explicit string sanitation"""
    string_cols = ["ticker", "company_name", "ceo_name", "sector"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper() if col == "ticker" else df[col].astype(str).str.strip()

    # Datetime coercion
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"].fillna("2026-10-31"), errors="coerce")
    return df

# Sub-issue #74: Strict t+1 execution lag enforcement.
def generate_universe_mask(
    prices: pd.DataFrame, 
    roster: pd.DataFrame, 
    lag_days: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Builds the point-in-time constituent eligibility matrix M_{t, i} with t+1 execution lag.
    
    Parameters:
        prices: Master adjusted prices DataFrame with SPY calendar index.
        roster: Audited executive tenure roster.
        lag_days: Look-ahead defense shift (default: 1 day for t+1 execution).
    
    Returns:
        mask_lagged: Boolean DataFrame (dates x tickers) where True = eligible to hold.
        active_counts: Daily series of active Black CEO count (N_t).
    """
    # Exclude benchmark columns from constituent universe
    constituent_tickers = [col for col in prices.columns if col not in ["SPY", "RSP"]]
    dates = prices.index

    # Initialize raw point-in-time tenure eligibility matrix
    raw_mask = pd.DataFrame(False, index=dates, columns=constituent_tickers)

    # Sub-task #76: Vectorized date broadcasting per roster entry
    for _, row in roster.iterrows():
        ticker = row["ticker"]
        if ticker in raw_mask.columns:
            start = row["start_date"]
            end = row["end_date"]
            
            # Tenure condition: start_date <= date <= end_date
            in_tenure = (dates >= start) & (dates <= end)
            
            # Asset must also have valid non-NaN price on date t
            has_price = prices[ticker].notna()
            
            # Combine tenure eligibility
            raw_mask[ticker] = raw_mask[ticker] | (in_tenure & has_price)

    # Sub-task #74: Enforce strict t+1 execution lag
    if lag_days > 0:
        mask_lagged = raw_mask.shift(lag_days).fillna(False).astype(bool)
        logging.info(f"Enforced strict t+{lag_days} execution lag on universe mask.")
    else:
        mask_lagged = raw_mask.copy()

    # Active count N_t for cash drag / diversification defense
    active_counts = mask_lagged.sum(axis=1)
    logging.info(f"Generated universe mask: {mask_lagged.shape} | Max active N_t: {active_counts.max()}")

    return mask_lagged, active_counts

def export_universe_mask(mask: pd.DataFrame, output_path: Path = UNIVERSE_MASK_PATH) -> Path:
    """Exports universe mask to data/universe_mask.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(output_path)
    logging.info(f"Saved universe mask to {output_path}")
    return output_path


if __name__ == "__main__":
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices matrix not found at {PRICES_PATH}. Run src.data_loader first.")
    
    df_prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    df_roster = load_clean_roster()
    mask, n_t = generate_universe_mask(df_prices, df_roster, lag_days=1)
    export_universe_mask(mask)
    
    print("\n--- SUB-TASKS #74-#76 VALIDATION ---")
    print(f"Mask Shape: {mask.shape}")
    print(f"Constituents Tracked: {list(mask.columns)}")
    print(f"Active Black CEO count N_t summary:")
    print(f"  Min: {n_t.min()} | Mean: {n_t.mean():.2f} | Max: {n_t.max()}")
    print("\nSample active constituents on 2026-10-30:")
    current_active = mask.loc["2026-10-30"]
    print(current_active[current_active].index.tolist())
