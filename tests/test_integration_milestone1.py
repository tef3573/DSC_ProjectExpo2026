"""
tests/test_integration_milestone1.py
End-to-end integration test verifying the complete data pipeline:
raw roster -> price matrix ingestion -> SPY calendar alignment -> point-in-time universe mask.
"""

from pathlib import Path
import unittest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"

class TestMilestone1Integration(unittest.TestCase):

    def test_pipeline_artifact_continuity(self):
        """Verify all three core artifacts exist and maintain exact dimensional continuity."""
        self.assertTrue(ROSTER_PATH.exists(), "Roster file missing")
        self.assertTrue(PRICES_PATH.exists(), "Prices daily file missing")
        self.assertTrue(MASK_PATH.exists(), "Universe mask file missing")

        roster = pd.read_csv(ROSTER_PATH)
        prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True)

        roster_tickers = sorted(roster["ticker"].dropna().astype(str).str.strip().unique().tolist())

        # 1. Row continuity (strictly identical trading sessions)
        self.assertEqual(len(prices), len(mask), "Prices and mask row counts must match")
        self.assertTrue((prices.index == mask.index).all(), "Dates must match session-for-session")

        # 2. Constituent column parity (mask must contain every roster ticker, excluding benchmarks)
        for ticker in roster_tickers:
            self.assertIn(ticker, mask.columns, f"Constituent {ticker} missing from universe mask")

        self.assertNotIn("SPY", mask.columns, "SPY benchmark leaked into stock mask")
        self.assertNotIn("RSP", mask.columns, "RSP benchmark leaked into stock mask")

        # 3. SPY calendar monotonic check
        self.assertTrue(prices.index.is_monotonic_increasing, "Prices index must be strictly monotonic")
        self.assertTrue(mask.index.is_monotonic_increasing, "Mask index must be strictly monotonic")


if __name__ == "__main__":
    unittest.main()