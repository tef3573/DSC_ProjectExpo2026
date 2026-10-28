"""
src/data_loader.py
Fetches daily adjusted close price matrices and Kenneth French 5-Factor risk-free series.
"""
from pathlib import Path
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DELISTED_DIR = DATA_DIR / "delisted"


def load_roster() -> pd.DataFrame:
    roster_path = DATA_DIR / "black_ceo_roster.csv"
    if not roster_path.exists():
        raise FileNotFoundError(f"Roster file not found: {roster_path}")
    df = pd.read_csv(roster_path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"].fillna("2026-10-27"))
    return df


def fetch_ticker_series(ticker: str, start: str, end: str) -> pd.Series:
    """Fetches daily Adjusted Close prices for a single ticker."""
    # Check local delisted archive first
    local_csv = DELISTED_DIR / f"{ticker}.csv"
    if local_csv.exists():
        logging.info(f"Loading {ticker} from local delisted archive: {local_csv}")
        df_delisted = pd.read_csv(local_csv, parse_dates=["Date"])
        df_delisted = df_delisted.set_index("Date").sort_index()
        col = "Adj Close" if "Adj Close" in df_delisted.columns else "Close"
        return df_delisted[col]

    # Fetch from yfinance
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty or "Adj Close" not in raw:
            logging.warning(f"No price data returned from yfinance for {ticker}")
            return pd.Series(dtype=float, name=ticker)
        
        series = raw["Adj Close"].squeeze()
        series.name = ticker
        return series
    except Exception as e:
        logging.error(f"Error downloading {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)


def build_price_matrix(start_date: str = "2000-01-01", end_date: str = "2026-10-27") -> pd.DataFrame:
    """Builds and exports the master prices_daily.csv matrix."""
    roster = load_roster()
    tickers = sorted(roster["ticker"].unique().tolist())
    benchmarks = ["SPY", "RSP"]
    all_symbols = list(dict.fromkeys(tickers + benchmarks))

    logging.info(f"Building price matrix for {len(all_symbols)} symbols...")
    
    price_dict = {}
    failed_tickers = []

    for sym in all_symbols:
        series = fetch_ticker_series(sym, start=start_date, end=end_date)
        if series.dropna().empty:
            failed_tickers.append(sym)
        else:
            price_dict[sym] = series

    if failed_tickers:
        logging.error(f"Failed to populate price history for: {failed_tickers}")

    # Combine all series on common market calendar
    prices_df = pd.DataFrame(price_dict).sort_index()
    
    # Trim to market trading days where benchmark SPY is available
    if "SPY" in prices_df.columns:
        spy_calendar = prices_df["SPY"].dropna().index
        prices_df = prices_df.reindex(spy_calendar)

    output_path = DATA_DIR / "prices_daily.csv"
    prices_df.to_csv(output_path)
    logging.info(f"Master price matrix saved to {output_path} (Shape: {prices_df.shape})")
    
    return prices_df


if __name__ == "__main__":
    build_price_matrix()