"""
tests/test_export.py
Unit tests verifying export artifacts.
- Sub-issue #127: Export daily time series to data/backtest_results.csv.
"""
from pathlib import Path
import unittest
import pandas as pd
from src.backtest import run_backtest, export_backtest_results, DEFAULT_EXPORT_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"


class TestArtifactExport(unittest.TestCase):

    def test_export_daily_time_series_file_and_columns(self):
        """
        Sub-issue #127: Verify data/backtest_results.csv is generated and contains required columns.
        """
        if not PRICES_PATH.exists() or not MASK_PATH.exists():
            self.skipTest("Missing raw price or universe artifacts")

        prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True).astype(bool)

        # Run backtest with export enabled
        res = run_backtest(prices, mask, export_results=True)

        self.assertTrue(DEFAULT_EXPORT_PATH.exists(), f"File not found: {DEFAULT_EXPORT_PATH}")
        exported_df = pd.read_csv(DEFAULT_EXPORT_PATH, index_col=0, parse_dates=True)

        required_cols = [
            "Black_CEO_Gross",
            "Black_CEO_Net",
            "Black_CEO_Cumulative",
            "Cash_Weight",
            "Active_Constituents_Nt",
            "One_Way_Turnover",
            "Transaction_Costs",
            "Strategy_Drawdown",
            "SPY_Daily",
            "SPY_Cumulative",
            "SPY_Drawdown",
            "RSP_Daily",
            "RSP_Cumulative",
            "RSP_Drawdown",
        ]

        for col in required_cols:
            self.assertIn(col, exported_df.columns, f"Missing required column: {col}")

        # Dimensional continuity
        self.assertEqual(len(exported_df), len(res))
        self.assertGreater(len(exported_df), 6000)
        self.assertFalse(exported_df["Black_CEO_Cumulative"].isna().any())


if __name__ == "__main__":
    unittest.main()
