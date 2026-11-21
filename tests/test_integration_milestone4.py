"""
tests/test_integration_milestone4.py
End-to-End Integration Tests for Milestone 4: Factor Attribution & Statistical Robustness.

Sub-issue #236: Validates the complete pipeline from strategy CSV inputs through FF5 OLS
regression, Newey-West HAC p-value tables, factor_loadings.csv export, and Monte Carlo
bootstrap mc_results.json export.
"""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.attribution import (
    BACKTEST_RESULTS_PATH,
    DATA_DIR,
    FF5_FACTOR_COLUMNS,
    FactorAttributionConfig,
    calculate_annualized_alpha_inference,
    download_ff5_daily_factors,
    estimate_ff5_ols_loadings,
    evaluate_alpha_hypothesis_test,
    export_factor_loadings_csv,
    format_ff5_regression_markdown_table,
)
from src.bootstrap import SyntheticBootstrapEngine, export_mc_results_json


class TestMilestone4IntegrationPipeline(unittest.TestCase):
    """End-to-end integration tests for the Milestone 4 attribution & bootstrap pipeline."""

    def test_ff5_regression_to_markdown_pvalue_table(self) -> None:
        """Sub-issue #236: Verify FF5 regression, HAC alpha inference, and p-value table."""
        self.assertTrue(BACKTEST_RESULTS_PATH.exists(), "Missing data/backtest_results.csv")
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(FactorAttributionConfig())

        ew_summary = estimate_ff5_ols_loadings(
            bt_df["Black_CEO_Net"],
            ff5_df,
            scheme_name="Equal-Weight (Net)",
        )
        alpha_inf = calculate_annualized_alpha_inference(
            bt_df["Black_CEO_Net"],
            ff5_df,
            scheme_name="Equal-Weight (Net)",
            hac_max_lags=5,
        )
        alpha_dict = evaluate_alpha_hypothesis_test(
            bt_df["Black_CEO_Net"],
            ff5_df,
            scheme_name="Equal-Weight (Net)",
            hac_max_lags=5,
        )

        self.assertAlmostEqual(
            alpha_inf.alpha_annualized,
            ew_summary.alpha_annualized,
            places=10,
        )
        self.assertTrue(0.0 < alpha_inf.p_value_one_sided_gt_zero < 0.10)
        self.assertIsInstance(alpha_dict, dict)

        md_table = format_ff5_regression_markdown_table([ew_summary])
        self.assertIn("Annualized Alpha", md_table)
        self.assertIn("$p$-value", md_table)
        for factor in FF5_FACTOR_COLUMNS:
            self.assertIn(factor, md_table)

    def test_artifact_exports_csv_and_bootstrap_json(self) -> None:
        """Sub-issue #236: Verify end-to-end export of factor_loadings.csv and mc_results.json."""
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        mask_df = pd.read_csv(DATA_DIR / "universe_mask.csv", index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(FactorAttributionConfig())

        scheme_columns = [
            ("Equal-Weight (Net)", "Black_CEO_Net"),
            ("Convex-Quality (Net)", "Convex_Quality_Net"),
        ]
        summaries = [
            estimate_ff5_ols_loadings(bt_df[col], ff5_df, scheme_name=label)
            for label, col in scheme_columns
            if col in bt_df.columns
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_out = Path(tmp_dir) / "factor_loadings.csv"
            json_out = Path(tmp_dir) / "mc_results.json"

            exported_df = export_factor_loadings_csv(summaries, output_path=csv_out)
            self.assertTrue(csv_out.exists())
            self.assertEqual(exported_df.shape[0], 7 * len(summaries))

            engine = SyntheticBootstrapEngine.from_sp_benchmark_series(
                benchmark_returns=bt_df["SPY_Daily"],
                active_counts=mask_df.sum(axis=1),
                pool_size=500,
                seed=42,
            )
            trials = engine.simulate_random_portfolios(n_trials=500, seed=42, n_jobs=-1)
            payload = export_mc_results_json(
                {label: bt_df[col] for label, col in scheme_columns if col in bt_df.columns},
                trials,
                output_path=json_out,
                seed=42,
                risk_free_annual=0.015,
            )
            loaded_json = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(loaded_json["metadata"]["seed"], 42)
            self.assertIn(
                "p_value_right_tailed",
                loaded_json["strategies"]["Equal-Weight (Net)"]["cagr"],
            )
            self.assertEqual(
                payload["strategies"]["Equal-Weight (Net)"]["sharpe"]["n_sessions"],
                bt_df["Black_CEO_Net"].dropna().shape[0],
            )


if __name__ == "__main__":
    unittest.main()
