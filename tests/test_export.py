"""
tests/test_export.py
Unit tests verifying export artifacts.
- Sub-issue #127: Export daily time series to data/backtest_results.csv.
- Sub-issue #128: Export summary metrics dictionary to data/summary_metrics.json.
"""
from pathlib import Path
import json
import unittest
import pandas as pd
from src.backtest import run_backtest, DEFAULT_EXPORT_PATH
from src.metrics import build_summary_metrics_dict, export_summary_metrics, DEFAULT_JSON_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"


class TestArtifactExport(unittest.TestCase):

    def test_export_daily_time_series_file_and_columns(self):
        """Sub-issue #127: Verify data/backtest_results.csv is generated and contains required columns."""
        if not PRICES_PATH.exists() or not MASK_PATH.exists():
            self.skipTest("Missing raw price or universe artifacts")

        prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True).astype(bool)

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
            self.assertIn(col, exported_df.columns)

        self.assertEqual(len(exported_df), len(res))
        self.assertGreater(len(exported_df), 6000)

    def test_export_summary_metrics_json_schema(self):
        """Sub-issue #128: Verify data/summary_metrics.json schema and required metrics."""
        if not DEFAULT_EXPORT_PATH.exists():
            self.skipTest("Missing backtest_results.csv artifact")

        results_df = pd.read_csv(DEFAULT_EXPORT_PATH, index_col=0, parse_dates=True)
        summary_dict = build_summary_metrics_dict(results_df, rf_annual=0.015)
        exported_path = export_summary_metrics(summary_dict)

        self.assertTrue(exported_path.exists(), f"File not found: {exported_path}")

        with open(exported_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # High-level section assertions
        self.assertIn("metadata", data)
        self.assertIn("strategy_performance", data)
        self.assertIn("benchmark_spy_full_horizon", data)
        self.assertIn("benchmark_rsp_isolated", data)

        # Strategy performance metric keys
        perf = data["strategy_performance"]
        expected_perf_keys = [
            "cagr_net",
            "cagr_gross",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "cumulative_multiple_net",
            "mean_daily_turnover",
            "total_transaction_friction_cost"
        ]
        for k in expected_perf_keys:
            self.assertIn(k, perf)
            self.assertIsInstance(perf[k], (float, int))

        # Benchmark comparison keys
        for b_key in ["benchmark_spy_full_horizon", "benchmark_rsp_isolated"]:
            bench_data = data[b_key]
            for metric in ["beta", "alpha_annual", "tracking_error", "strategy_cagr", "benchmark_cagr"]:
                self.assertIn(metric, bench_data)


if __name__ == "__main__":
    unittest.main()
