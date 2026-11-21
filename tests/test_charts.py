"""
tests/test_charts.py
Unit and serialization tests for Milestone 5 interactive Plotly visualizations (#247).
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import plotly.graph_objects as go

from src.charts import (
    build_active_constituent_count_figure,
    build_cumulative_equity_curve_figure,
    build_underwater_drawdown_figure,
    export_figure_json,
    validate_plotly_json_schema,
)


def _assert_all_numbers_finite(obj: object) -> None:
    """Recursively assert that no NaN or Inf floats exist in serialized JSON payloads."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise AssertionError(f"Non-finite float encountered in Plotly JSON: {obj}")
    elif isinstance(obj, list):
        for item in obj:
            _assert_all_numbers_finite(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            _assert_all_numbers_finite(value)


class TestInteractivePlotlyCharts(unittest.TestCase):
    """Sub-issue #247: Verify Plotly figure builders and deterministic JSON serialization."""

    def test_cumulative_equity_curve_figure_and_json_serialization(self) -> None:
        """Verify cumulative equity curve builds traces, controls, and valid JSON."""
        fig = build_cumulative_equity_curve_figure()
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 4)
        self.assertEqual(len(fig.layout.updatemenus), 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = export_figure_json(fig, Path(tmp_dir) / "equity.json")
            self.assertTrue(out_path.exists())
            parsed = json.loads(out_path.read_text(encoding="utf-8"))
            validate_plotly_json_schema(parsed)
            _assert_all_numbers_finite(parsed)

    def test_underwater_drawdown_figure_and_json_serialization(self) -> None:
        """Verify underwater drawdown figure uses non-positive percentages and valid JSON."""
        fig = build_underwater_drawdown_figure()
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 4)
        for trace in fig.data:
            self.assertTrue(all(val <= 0.0 for val in trace.y))

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = export_figure_json(fig, Path(tmp_dir) / "drawdown.json")
            parsed = json.loads(out_path.read_text(encoding="utf-8"))
            validate_plotly_json_schema(parsed)
            _assert_all_numbers_finite(parsed)

    def test_active_constituent_count_figure_and_json_serialization(self) -> None:
        """Verify active constituent count area figure serializes cleanly to JSON."""
        fig = build_active_constituent_count_figure()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].fill, "tozeroy")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = export_figure_json(fig, Path(tmp_dir) / "breadth.json")
            parsed = json.loads(out_path.read_text(encoding="utf-8"))
            validate_plotly_json_schema(parsed)
            _assert_all_numbers_finite(parsed)


if __name__ == "__main__":
    unittest.main()
