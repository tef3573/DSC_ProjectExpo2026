"""
tests/test_backtest.py
Unit tests verifying return calculation, cash-drag concentration defense, and weight drift.
- Sub-issue #104: Assert cash plus equity allocation weights sum to exactly 1.0.
- Sub-issue #106: Calculate inter-period weight drift from asset price returns.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.backtest import compute_equal_weights, compute_returns, compute_weight_drift, run_backtest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"


class TestBacktestEngine(unittest.TestCase):

    def test_compute_returns_integrity(self):
        """Verify arithmetic percentage return calculation."""
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

    def test_cash_plus_equity_sums_to_unity(self):
        """Sub-issue #104: Assert cash plus equity allocation weights sum to exactly 1.0."""
        test_mask = pd.DataFrame({
            "A": [False, True,  True,  True,  True],
            "B": [False, False, True,  True,  True],
            "C": [False, False, False, True,  True],
            "D": [False, False, False, False, True],
        })
        stock_w, cash_w, n_t = compute_equal_weights(test_mask, min_diversification=3)

        total_weights = stock_w.sum(axis=1) + cash_w
        np.testing.assert_allclose(
            total_weights.values,
            np.ones(len(total_weights)),
            rtol=1e-6,
            err_msg="Cash plus equity weights did not sum to 1.0 across all regimes"
        )

    def test_proportional_scaling_when_nt_less_than_three(self):
        """Sub-issue #100: Validate equity capping at 1/3 when Nt < 3."""
        test_mask = pd.DataFrame({
            "A": [True,  True,  True],
            "B": [False, True,  True],
            "C": [False, False, True]
        })
        stock_w, cash_w, n_t = compute_equal_weights(test_mask, min_diversification=3)

        # Session 0: Nt = 1 -> Stock A = 1/3, Cash = 2/3
        self.assertAlmostEqual(stock_w.loc[0, "A"], 1.0 / 3.0)
        self.assertAlmostEqual(cash_w.loc[0], 2.0 / 3.0)

        # Session 1: Nt = 2 -> Stock A = 1/3, Stock B = 1/3, Cash = 1/3
        self.assertAlmostEqual(stock_w.loc[1, "A"], 1.0 / 3.0)
        self.assertAlmostEqual(stock_w.loc[1, "B"], 1.0 / 3.0)
        self.assertAlmostEqual(cash_w.loc[1], 1.0 / 3.0)

        # Session 2: Nt = 3 -> Stock A = 1/3, Stock B = 1/3, Stock C = 1/3, Cash = 0.0
        self.assertAlmostEqual(stock_w.loc[2, "A"], 1.0 / 3.0)
        self.assertAlmostEqual(stock_w.loc[2, "B"], 1.0 / 3.0)
        self.assertAlmostEqual(stock_w.loc[2, "C"], 1.0 / 3.0)
        self.assertAlmostEqual(cash_w.loc[2], 0.0)

    def test_run_backtest_cash_drag_effect(self):
        """Sub-issue #101: Test portfolio return incorporates rf on zero active dates."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        test_prices = pd.DataFrame({
            "A": [100.0, 100.0, 100.0],
            "SPY": [300.0, 301.0, 302.0],
            "RSP": [100.0, 100.0, 100.0],
        }, index=dates)
        test_mask = pd.DataFrame({"A": [False, False, False]}, index=dates)

        rf_annual = 0.0252
        perf = run_backtest(test_prices, test_mask, rf_annual=rf_annual)
        expected_daily_rf = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

        self.assertAlmostEqual(perf["Black_CEO_EW"].iloc[1], expected_daily_rf, places=5)
        self.assertAlmostEqual(perf["Cash_Weight"].iloc[1], 1.0)

    def test_weight_drift_calculation(self):
        """Sub-issue #106: Test inter-period weight drift from asset price returns."""
        weights = pd.DataFrame({"A": [0.5], "B": [0.5]})
        asset_returns = pd.DataFrame({"A": [1.0], "B": [0.0]})
        portfolio_returns = pd.Series([0.5])

        drifted = compute_weight_drift(weights, asset_returns, portfolio_returns)

        # 0.5 * (1 + 1.0) / (1 + 0.5) = 1.0 / 1.5 = 0.6667
        # 0.5 * (1 + 0.0) / (1 + 0.5) = 0.5 / 1.5 = 0.3333
        self.assertAlmostEqual(drifted.loc[0, "A"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(drifted.loc[0, "B"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(drifted.sum(axis=1).iloc[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
