"""
tests/test_backtest.py
Unit tests verifying return calculation, cash-drag defense, weight drift, turnover friction,
scorecard formatting, and clean end-to-end simulated data backtests.
- Sub-issues #104, #106, #107, #108, #110, #129
- Sub-issue #134: Execute full test suite on clean simulated data.
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
    format_scorecard_table,
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

    def test_proportional_scaling_concentration_defense(self):
        """Sub-issue #100: Validate equity capping at 1/3 when Nt < 3."""
        test_mask = pd.DataFrame({
            "A": [True,  True,  True],
            "B": [False, True,  True],
            "C": [False, False, True]
        })
        stock_w, cash_w, n_t = compute_equal_weights(test_mask, min_diversification=3)

        self.assertAlmostEqual(stock_w.loc[0, "A"], 1.0 / 3.0)
        self.assertAlmostEqual(cash_w.loc[0], 2.0 / 3.0)

        self.assertAlmostEqual(stock_w.loc[1, "A"], 1.0 / 3.0)
        self.assertAlmostEqual(stock_w.loc[1, "B"], 1.0 / 3.0)
        self.assertAlmostEqual(cash_w.loc[1], 1.0 / 3.0)

        self.assertAlmostEqual(stock_w.loc[2, "A"], 1.0 / 3.0)
        self.assertAlmostEqual(stock_w.loc[2, "B"], 1.0 / 3.0)
        self.assertAlmostEqual(stock_w.loc[2, "C"], 1.0 / 3.0)
        self.assertAlmostEqual(cash_w.loc[2], 0.0)

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
        self.assertEqual(turnover.iloc[0], 0.0)
        self.assertEqual(costs.iloc[0], 0.0)

    def test_net_return_strictly_lte_gross_return(self):
        """Sub-issue #110: Verify net return is strictly <= gross return for all sessions."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        test_prices = pd.DataFrame({
            "A": [100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0],
            "B": [50.0, 49.0, 52.0, 51.0, 53.0, 52.0, 55.0, 54.0, 56.0, 57.0],
            "SPY": [300.0] * 10,
            "RSP": [100.0] * 10,
        }, index=dates)

        test_mask = pd.DataFrame({
            "A": [True, False, True, False, True, False, True, False, True, False],
            "B": [False, True, False, True, False, True, False, True, False, True],
        }, index=dates)

        perf = run_backtest(test_prices, test_mask, fee_bps=10.0, export_results=False)

        diff = perf["Black_CEO_Gross"] - perf["Black_CEO_Net"]
        self.assertTrue((diff >= -1e-9).all())
        self.assertTrue((perf["Transaction_Costs"] >= 0.0).all())

    def test_format_scorecard_table_structure(self):
        """Sub-issue #129: Verify format_scorecard_table produces complete aligned table."""
        mock_summary = {
            "metadata": {
                "backtest_start": "2000-01-03",
                "backtest_end": "2026-10-30",
                "trading_sessions": 6700
            },
            "strategy_performance": {
                "cagr_net": 0.125,
                "annualized_volatility": 0.182,
                "sharpe_ratio": 0.65,
                "sortino_ratio": 0.88,
                "calmar_ratio": 0.35,
                "max_drawdown": 0.35,
                "cumulative_multiple_net": 15.2,
                "annualized_turnover": 0.45,
                "total_transaction_friction_cost": 0.08
            },
            "benchmark_spy_full_horizon": {
                "benchmark_cagr": 0.085,
                "benchmark_volatility": 0.151,
                "benchmark_sharpe": 0.48,
                "benchmark_mdd": 0.55,
                "beta": 0.95,
                "alpha_annual": 0.035,
                "tracking_error": 0.075
            },
            "benchmark_rsp_isolated": {
                "benchmark_cagr": 0.102,
                "benchmark_volatility": 0.178,
                "benchmark_sharpe": 0.52,
                "benchmark_mdd": 0.58,
                "beta": 0.98,
                "alpha_annual": 0.021,
                "tracking_error": 0.062
            }
        }
        table_str = format_scorecard_table(mock_summary)
        self.assertIn("BLACK CEO PORTFOLIO: INSTITUTIONAL PERFORMANCE SCORECARD", table_str)
        self.assertIn("Sharpe Ratio", table_str)
        self.assertIn("SPY (Cap-Wtd)", table_str)
        self.assertIn("RSP (Eq-Wtd)*", table_str)
        self.assertIn("Cumulative Wealth Multiple", table_str)

    def test_full_simulation_on_synthetic_data(self):
        """
        Sub-issue #134: Execute end-to-end backtest on purely synthetic geometric random walks
        with transition regimes (0, 1, 2, 4 active assets) to guarantee mathematical invariants.
        """
        np.random.seed(1337)
        sessions = 500
        dates = pd.date_range("2010-01-01", periods=sessions, freq="B")
        assets = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]

        price_dict = {}
        for a in assets:
            daily_returns = np.random.normal(loc=0.0004, scale=0.015, size=sessions)
            price_dict[a] = 100.0 * np.exp(np.cumsum(daily_returns))
        price_dict["SPY"] = 200.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, size=sessions)))
        price_dict["RSP"] = 80.0 * np.exp(np.cumsum(np.random.normal(0.00035, 0.011, size=sessions)))
        synth_prices = pd.DataFrame(price_dict, index=dates)

        synth_mask = pd.DataFrame(False, index=dates, columns=assets)
        synth_mask.iloc[50:150, 0] = True
        synth_mask.iloc[150:300, [0, 1]] = True
        synth_mask.iloc[300:, :] = True

        res = run_backtest(synth_prices, synth_mask, fee_bps=10.0, rf_annual=0.02, export_results=False)

        self.assertEqual(len(res), sessions)
        self.assertTrue((res.iloc[0:50]["Cash_Weight"] == 1.0).all())
        self.assertTrue(np.allclose(res.iloc[50:150]["Cash_Weight"], 2.0 / 3.0))
        self.assertTrue(np.allclose(res.iloc[150:300]["Cash_Weight"], 1.0 / 3.0))
        self.assertTrue(np.allclose(res.iloc[300:]["Cash_Weight"], 0.0))
        self.assertTrue((res["Strategy_Drawdown"] <= 1e-9).all())
        self.assertFalse(res["Black_CEO_Cumulative"].isna().any())


if __name__ == "__main__":
    unittest.main()
