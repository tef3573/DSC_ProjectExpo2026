"""
Unit and integration tests for Fama-French 5-Factor ingestion, decimal conversion,
and calendar alignment (Milestone 4, Issue #196 / Sub-issue #207).
"""
import unittest
import numpy as np
import pandas as pd
from src.attribution import (
    BACKTEST_RESULTS_PATH,
    FF5_ALL_COLUMNS,
    FF5_CACHE_PATH,
    align_strategy_and_factors,
    convert_percent_to_decimal,
    download_ff5_daily_factors,
)


class TestFactorIngestionAndAlignment(unittest.TestCase):
    """Test suite verifying FF5 data normalization and strategy calendar alignment."""

    def test_convert_percent_to_decimal_accuracy(self) -> None:
        """Sub-issue #205: Verify percentage figures divide by 100.0 accurately."""
        raw_df = pd.DataFrame(
            {
                "Mkt-RF": [1.25, -0.82],
                "SMB": [0.40, 0.10],
                "HML": [-0.50, 0.00],
                "RMW": [0.68, -0.35],
                "CMA": [0.39, 0.19],
                "RF": [0.01, 0.02],
            },
            index=pd.to_datetime(["2026-09-28", "2026-09-30"]),
        )
        dec_df = convert_percent_to_decimal(raw_df)
        self.assertAlmostEqual(float(dec_df.loc["2026-09-28", "Mkt-RF"]), 0.0125, places=8)
        self.assertAlmostEqual(float(dec_df.loc["2026-09-30", "Mkt-RF"]), -0.0082, places=8)
        self.assertAlmostEqual(float(dec_df.loc["2026-09-28", "RF"]), 0.0001, places=8)

    def test_cached_ff5_factors_are_decimal_scaled(self) -> None:
        """Sub-issue #204 & #205: Verify cached FF5 dataset exists and uses decimal units."""
        self.assertTrue(FF5_CACHE_PATH.exists(), f"Missing FF5 cache at {FF5_CACHE_PATH}")
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)
        self.assertEqual(list(ff5_df.columns), list(FF5_ALL_COLUMNS))
        self.assertGreater(len(ff5_df), 6000)
        self.assertLess(float(ff5_df.abs().max().max()), 0.25)

    def test_date_alignment_between_strategy_and_ff5_factors(self) -> None:
        """Sub-issue #207: Verify exact DatetimeIndex alignment and excess return identity."""
        self.assertTrue(
            BACKTEST_RESULTS_PATH.exists(),
            f"Missing backtest results at {BACKTEST_RESULTS_PATH}",
        )
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)

        for scheme_col in ("Black_CEO_Net", "Convex_Quality_Net"):
            with self.subTest(scheme=scheme_col):
                excess_ret, aligned_ff5 = align_strategy_and_factors(
                    bt_df[scheme_col],
                    ff5_df,
                )
                # Indices must match identically and have zero NaNs
                self.assertTrue(excess_ret.index.equals(aligned_ff5.index))
                self.assertFalse(excess_ret.isna().any())
                self.assertFalse(aligned_ff5.isna().any().any())
                self.assertGreater(len(excess_ret), 6000)

                # Verify R_e = r_net - RF on every aligned session
                expected_excess = (
                    bt_df.loc[excess_ret.index, scheme_col] - aligned_ff5["RF"]
                )
                np.testing.assert_allclose(
                    excess_ret.to_numpy(),
                    expected_excess.to_numpy(),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_alignment_handles_timezone_and_non_overlapping_dates(self) -> None:
        """Sub-issue #207: Verify tz-aware normalization and ValueError on zero overlap."""
        dates_ff5 = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        ff5_mock = pd.DataFrame(
            {col: [0.001, -0.002, 0.003] for col in FF5_ALL_COLUMNS},
            index=dates_ff5,
        )
        # Subset overlap with tz-aware index
        strat_tz = pd.Series(
            [0.01, -0.005],
            index=pd.to_datetime(["2020-01-03", "2020-01-06"]).tz_localize("UTC"),
        )
        excess_ret, _ = align_strategy_and_factors(strat_tz, ff5_mock)
        self.assertEqual(len(excess_ret), 2)
        self.assertIsNone(excess_ret.index.tz)
        self.assertAlmostEqual(float(excess_ret.iloc[0]), 0.01 - (-0.002), places=12)
        # Disjoint dates must raise ValueError
        strat_disjoint = pd.Series([0.01], index=pd.to_datetime(["1995-01-03"]))
        with self.assertRaises(ValueError):
            align_strategy_and_factors(strat_disjoint, ff5_mock)


if __name__ == "__main__":
    unittest.main()
