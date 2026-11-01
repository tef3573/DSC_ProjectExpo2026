"""
tests/test_data_schema.py
Unit tests to validate the integrity, schema, and point-in-time date consistency
of the executive roster and delisted price archives; 
and final shape of prices_daily.csv.
"""

from pathlib import Path
import unittest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
JCP_PATH = DATA_DIR / "delisted" / "JCP.csv"
PRICES_PATH = DATA_DIR / "prices_daily.csv"

class TestDataSchema(unittest.TestCase):

    def test_roster_file_and_columns(self):
        """Verify black_ceo_roster.csv exists and matches expected schema."""
        self.assertTrue(ROSTER_PATH.exists(), f"Roster file missing at {ROSTER_PATH}")
        df = pd.read_csv(ROSTER_PATH)
        df.columns = df.columns.str.strip()

        required_cols = {"ticker", "company_name", "ceo_name", "start_date", "end_date", "sector"}
        self.assertTrue(
            required_cols.issubset(set(df.columns)),
            f"Missing required columns. Found: {list(df.columns)}"
        )
        self.assertGreaterEqual(len(df), 20, f"Expected at least 20 roster entries, found {len(df)}")

    def test_roster_date_consistency(self):
        """Verify all start dates precede end dates and parse as valid timestamps."""
        df = pd.read_csv(ROSTER_PATH)
        df.columns = df.columns.str.strip()

        start_dates = pd.to_datetime(df["start_date"], errors="coerce")
        end_dates = pd.to_datetime(df["end_date"].fillna("2026-10-31"), errors="coerce")

        self.assertFalse(start_dates.isna().any(), "Found invalid or unparseable start_date entries")
        self.assertFalse(end_dates.isna().any(), "Found invalid or unparseable end_date entries")
        self.assertTrue((start_dates <= end_dates).all(), "start_date must be earlier than or equal to end_date")

    def test_delisted_jcp_schema(self):
        """Verify data/delisted/JCP.csv contains clean dates and positive closing prices."""
        self.assertTrue(JCP_PATH.exists(), f"Delisted JCP archive missing at {JCP_PATH}")
        df = pd.read_csv(JCP_PATH)
        df.columns = df.columns.str.strip()

        self.assertIn("Date", df.columns, "Date column missing from JCP.csv")
        self.assertIn("Adj Close", df.columns, "Adj Close column missing from JCP.csv")

        dates = pd.to_datetime(df["Date"], errors="coerce")
        prices = pd.to_numeric(df["Adj Close"], errors="coerce")

        self.assertFalse(dates.isna().any(), "Found invalid Date entries in JCP.csv")
        self.assertFalse(prices.isna().any(), "Found non-numeric Adj Close values in JCP.csv")
        self.assertTrue((prices > 0).all(), "All closing prices must be strictly positive")
        self.assertGreaterEqual(len(df), 700, f"Expected >=700 NYSE trading days for JCP, found {len(df)}")

    def test_prices_daily_shape_and_benchmarks(self):
            """Sub-task #68: Validate prices_daily.csv exists, has >=6700 dates, and contains benchmarks."""
            self.assertTrue(PRICES_PATH.exists(), f"prices_daily.csv missing at {PRICES_PATH}")
            df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

            # Minimum required trading sessions across 2000-2026
            self.assertGreaterEqual(df.shape[0], 6700, f"Expected >=6700 rows, found {df.shape[0]}")
            # 21 constituents + SPY + RSP = at least 23 columns
            self.assertGreaterEqual(df.shape[1], 23, f"Expected >=23 columns, found {df.shape[1]}")

            # Benchmark verification
            self.assertIn("SPY", df.columns, "SPY benchmark missing from master price matrix")
            self.assertIn("RSP", df.columns, "RSP benchmark missing from master price matrix")

            # SPY prices must never be NaN across the calendar
            self.assertEqual(df["SPY"].isna().sum(), 0, "SPY benchmark has missing values in trading calendar")


if __name__ == "__main__":
    unittest.main()
