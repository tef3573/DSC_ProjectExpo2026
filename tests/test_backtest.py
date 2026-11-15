"""
tests/test_backtest.py
Unit tests verifying return calculation, cash-drag defense, weight drift, turnover friction,
scorecard formatting, comparative scheme integration, and stress-window stability.
- Sub-issues #104, #106, #107, #108, #110, #129, #134
- Sub-issues #179, #180, #181, #183
- Sub-issue #191: Verify optimizer performance stability under stress windows.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.backtest import (
    PRICES_PATH,
    MASK_PATH,
    compute_equal_weights,
    compute_returns,
    compute_weight_drift,
    compute_turnover_and_costs,
    format_scorecard_table,
    format_comparative_scorecard,
    run_backtest,
    run_comparative_backtests,
)


class TestBacktestEngine(unittest.TestCase):
    """Core unit tests for backtest mechanics and institutional scorecard formatting."""

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
        stock_w, cash_w, _ = compute_equal_weights(test_mask, min_diversification=3)

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
        stock_w, cash_w, _ = compute_equal_weights(test_mask, min_diversification=3)

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


class TestRebalancingIntegrationAndFriction(unittest.TestCase):
    """Sub-issues #179, #180, #181, #183: Multi-scheme integration and turnover tests."""

    @staticmethod
    def _build_synthetic_universe(sessions: int = 260, seed: int = 42):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-04", periods=sessions)
        assets = ["A1", "A2", "A3", "A4", "A5"]
        price_dict = {
            a: 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, size=sessions)))
            for a in assets
        }
        price_dict["SPY"] = 350.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, size=sessions)))
        price_dict["RSP"] = 130.0 * np.exp(np.cumsum(rng.normal(0.00035, 0.011, size=sessions)))
        prices = pd.DataFrame(price_dict, index=dates)
        mask = pd.DataFrame(True, index=dates, columns=assets)
        return prices, mask

    def test_comparative_backtests_all_three_schemes(self):
        """Sub-issues #179 & #180: Verify comparative runner outputs valid curves for all 3 schemes."""
        prices, mask = self._build_synthetic_universe()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp_path = Path(tmpdir) / "comparative_schemes.csv"
            comp_df, summaries = run_comparative_backtests(
                prices=prices,
                mask=mask,
                export_comparative=True,
                export_path=comp_path,
            )
            self.assertTrue(comp_path.exists())
            self.assertEqual(len(comp_df), len(prices))
            self.assertEqual(set(summaries.keys()), {"equal_weight", "inverse_vol", "convex_quality"})
            table = format_comparative_scorecard(summaries)
            self.assertIn("Convex Quality", table)

    def test_synchronized_monthly_rebalance_schedule_reduces_turnover(self):
        """Sub-issue #181: Monthly calendar synchronization eliminates intra-month churn."""
        prices, mask = self._build_synthetic_universe(sessions=260, seed=99)
        res_monthly = run_backtest(
            prices, mask, export_results=False, allocation_scheme="convex_quality", rebalance_monthly=True
        )
        res_daily = run_backtest(
            prices, mask, export_results=False, allocation_scheme="convex_quality", rebalance_monthly=False
        )
        self.assertLess(res_monthly["One_Way_Turnover"].sum(), res_daily["One_Way_Turnover"].sum())

    def test_optimized_portfolio_turnover_satisfies_friction_constraints(self):
        """Sub-issue #183: Verify convex_quality turnover and transaction costs stay bounded."""
        prices, mask = self._build_synthetic_universe(sessions=260, seed=7)
        res = run_backtest(
            prices, mask, fee_bps=10.0, export_results=False, allocation_scheme="convex_quality"
        )
        ann_turnover = float(res["One_Way_Turnover"].mean() * 252.0)
        self.assertLess(ann_turnover, 6.0)
        self.assertTrue((res["Transaction_Costs"] <= 0.001 + 1e-9).all())


class TestOptimizerStressWindowStability(unittest.TestCase):
    """Sub-issue #191: Integration tests verifying optimizer stability under stress windows."""

    def test_historical_crisis_windows_stability_and_vol_compression(self):
        """Verify convex_quality maintains finite weights, budget unity, and vol compression in 2008 & 2020."""
        self.assertTrue(PRICES_PATH.exists(), f"Missing {PRICES_PATH}")
        self.assertTrue(MASK_PATH.exists(), f"Missing {MASK_PATH}")

        prices_df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        mask_df = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True).astype(bool)

        stress_windows = [
            ("2008_GFC", "2008-06-01", "2009-06-30"),
            ("2020_COVID", "2020-01-02", "2020-06-30"),
        ]

        for label, start_dt, end_dt in stress_windows:
            p_slice = prices_df.loc[start_dt:end_dt]
            m_slice = mask_df.loc[start_dt:end_dt]

            res_cq = run_backtest(
                prices=p_slice,
                mask=m_slice,
                export_results=False,
                allocation_scheme="convex_quality",
                rebalance_monthly=True,
            )
            res_ew = run_backtest(
                prices=p_slice,
                mask=m_slice,
                export_results=False,
                allocation_scheme="equal_weight",
                rebalance_monthly=True,
            )

            self.assertFalse(res_cq.isna().any().any(), f"NaN detected in {label} stress window")
            self.assertTrue(np.all(np.isfinite(res_cq.values)), f"Inf detected in {label} stress window")
            self.assertTrue((res_cq["One_Way_Turnover"] <= 1.0 + 1e-8).all())
            self.assertTrue((res_cq["Black_CEO_Cumulative"] > 0.0).all())
            self.assertLess(float(res_cq["Black_CEO_Net"].std()), 0.10)
            self.assertLessEqual(
                float(res_cq["Black_CEO_Net"].std()),
                float(res_ew["Black_CEO_Net"].std()) * 1.05,
            )

    def test_synthetic_correlation_spike_and_universe_contraction_stress(self):
        """Verify graceful handling of 5x vol shock, near-collinear crash, and sparse contraction (Nt: 6 -> 2)."""
        rng = np.random.default_rng(191)
        n_days = 180
        n_assets = 6
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        tickers = [f"STRESS_{i}" for i in range(n_assets)]

        calm_rets = rng.normal(0.0004, 0.01, size=(90, n_assets))
        common_shock = rng.normal(-0.004, 0.045, size=(90, 1))
        idio_noise = rng.normal(0.0, 0.012, size=(90, n_assets))
        crisis_rets = common_shock + idio_noise

        all_rets = np.vstack([calm_rets, crisis_rets])
        prices_data = 100.0 * np.exp(np.cumsum(all_rets, axis=0))
        prices = pd.DataFrame(prices_data, index=dates, columns=tickers)
        prices["SPY"] = 400.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, size=n_days)))
        prices["RSP"] = 150.0 * np.exp(np.cumsum(rng.normal(0.0, 0.016, size=n_days)))

        mask = pd.DataFrame(True, index=dates, columns=tickers)
        mask.iloc[120:, 2:] = False

        res = run_backtest(
            prices=prices,
            mask=mask,
            export_results=False,
            allocation_scheme="convex_quality",
            rebalance_monthly=True,
        )

        self.assertFalse(res.isna().any().any())
        self.assertTrue(np.all(np.isfinite(res.values)))
        self.assertTrue((res.iloc[120:]["Cash_Weight"] >= (1.0 / 3.0) - 1e-6).all())


if __name__ == "__main__":
    unittest.main()
