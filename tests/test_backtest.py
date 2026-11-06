"""
tests/test_backtest.py
Unit tests verifying return calculation, cash-drag defense, weight drift, and turnover friction.
- Sub-issue #104: Assert cash plus equity allocation weights sum to exactly 1.0.
- Sub-issue #106: Calculate inter-period weight drift from asset price returns.
- Sub-issue #107: Deduct 10 bps transaction fee on one-way turnover.
- Sub-issue #108: Adjust turnover formulation for initial position inception.
- Sub-issue #110: Verify net return is strictly less than or equal to gross return.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.backtest import (
    compute_equal_weights,
    compute_returns,
    compute_weight_drift,
    compute_turnover_and_costs,
    run_backtest
)

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
            rtol=1e-6
        )

    def test_weight_drift_calculation(self):
        """Sub-issue #106: Test inter-period weight drift from asset price returns."""
        weights = pd.DataFrame({"A": [0.5], "B": [0.5]})
        asset_returns = pd.DataFrame({"A": [1.0], "B": [0.0]})
        portfolio_returns = pd.Series([0.5])

        drifted = compute_weight_drift(weights, asset_returns, portfolio_returns)
        self.assertAlmostEqual(drifted.loc[0, "A"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(drifted.loc[0, "B"], 1.0 / 3.0, places=4)

    def test_deduct_ten_bps_transaction_fee(self):
        """Sub-issue #107: Deduct exactly 10 bps (0.0010) per 100% one-way turnover."""
        target = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
        drifted = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})

        turnover, costs = compute_turnover_and_costs(target, drifted, fee_bps=10.0)
        self.assertAlmostEqual(turnover.iloc[1], 1.0, places=5)
        self.assertAlmostEqual(costs.iloc[1], 0.0010, places=5)

    def test_initial_inception_turnover_is_zero(self):
        """Sub-issue #108: Verify initial position inception at t=0 has zero turnover and fee."""
        target = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
        drifted = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})

        turnover, costs = compute_turnover_and_costs(target, drifted, fee_bps=10.0)
        self.assertEqual(turnover.iloc[0], 0.0, "Day 0 turnover must be exactly 0.0")
        self.assertEqual(costs.iloc[0], 0.0, "Day 0 transaction costs must be exactly 0.0")

    def test_net_return_strictly_lte_gross_return(self):
        """Sub-issue #110: Verify net return is strictly <= gross return for all sessions."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        test_prices = pd.DataFrame({
            "A": [100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0],
            "B": [50.0, 49.0, 52.0, 51.0, 53.0, 52.0, 55.0, 54.0, 56.0, 57.0],
            "SPY": [300.0] * 10,
            "RSP": [100.0] * 10,
        }, index=dates)

        # Alternating weights force regular non-zero turnover
        test_mask = pd.DataFrame({
            "A": [True, False, True, False, True, False, True, False, True, False],
            "B": [False, True, False, True, False, True, False, True, False, True],
        }, index=dates)

        perf = run_backtest(test_prices, test_mask, fee_bps=10.0)

        # R_net <= R_gross for every single session
        diff = perf["Black_CEO_Gross"] - perf["Black_CEO_Net"]
        self.assertTrue((diff >= -1e-9).all(), "Net returns exceeded gross returns on some dates")

        # Total transaction costs must be non-negative
        self.assertTrue((perf["Transaction_Costs"] >= 0.0).all())

        # Cumulative net return must be strictly less than gross return after trading
        gross_cum = (1.0 + perf["Black_CEO_Gross"]).cumprod().iloc[-1]
        net_cum = perf["Black_CEO_Cumulative"].iloc[-1]
        self.assertLess(net_cum, gross_cum, "Cumulative net wealth should be strictly less than gross wealth")


if __name__ == "__main__":
    unittest.main()
