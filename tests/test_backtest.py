"""
tests/test_backtest.py
Unit tests verifying return calculation, equal-weighting, and single-asset boundary cases.
- Sub-issue #98: Test portfolio gross returns against single-asset boundary cases.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.backtest import compute_equal_weights, compute_returns, run_backtest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"


class TestBacktestEngine(unittest.TestCase):

    def test_single_asset_boundary_case(self):
        """
        Sub-issue #98: When Nt == 1 for an asset, portfolio return must match that asset's return.
        """
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        test_prices = pd.DataFrame({
            "ABC": [100.0, 105.0, 102.0, 110.0, 115.0],
            "XYZ": [50.0, 50.0, 50.0, 50.0, 50.0],
            "SPY": [300.0, 303.0, 301.0, 305.0, 310.0],
            "RSP": [100.0, 101.0, 100.5, 102.0, 103.0],
        }, index=dates)

        test_mask = pd.DataFrame({
            "ABC": [True, True, True, True, True],
            "XYZ": [False, False, False, False, False],
        }, index=dates)

        perf = run_backtest(test_prices, test_mask)
        expected_returns = test_prices["ABC"].pct_change().fillna(0.0)

        # Match strategy column dynamically across naming variants
        strat_col = next((c for c in perf.columns if "Black_CEO" in c or "Strategy" in c or "Portfolio" in c), perf.columns[0])

        np.testing.assert_allclose(
            perf[strat_col].values,
            expected_returns.values,
            rtol=1e-5,
            err_msg="Single asset portfolio return did not equal constituent return"
        )

    def test_compute_returns_integrity(self):
        """Verify compute_returns computes arithmetic percentage changes cleanly."""
        test_prices = pd.DataFrame({
            "A": [100.0, 110.0, 99.0],
            "B": [50.0, 50.0, 55.0]
        })
        rets = compute_returns(test_prices)
        self.assertEqual(len(rets), len(test_prices))
        self.assertAlmostEqual(rets.iloc[1]["A"], 0.10)
        self.assertAlmostEqual(rets.iloc[2]["A"], -0.10)
        self.assertAlmostEqual(rets.iloc[1]["B"], 0.0)
        self.assertAlmostEqual(rets.iloc[2]["B"], 0.10)

    def test_equal_weights_properties(self):
        """Verify compute_equal_weights handles active constituent scaling."""
        test_mask = pd.DataFrame({
            "A": [True, False, False],
            "B": [True, True, False]
        })
        weights_res = compute_equal_weights(test_mask)

        # Unpack tuple or DataFrame return
        weights = weights_res[0] if isinstance(weights_res, tuple) else weights_res

        # Day 0: 2 active -> 0.5 each
        self.assertAlmostEqual(weights.loc[0, "A"], 0.5)
        self.assertAlmostEqual(weights.loc[0, "B"], 0.5)

        # Day 1: 1 active -> 1.0 for B, 0.0 for A
        self.assertAlmostEqual(weights.loc[1, "A"], 0.0)
        self.assertAlmostEqual(weights.loc[1, "B"], 1.0)

        # Day 2: 0 active -> 0.0 each
        self.assertAlmostEqual(weights.loc[2, "A"], 0.0)
        self.assertAlmostEqual(weights.loc[2, "B"], 0.0)


if __name__ == "__main__":
    unittest.main()
