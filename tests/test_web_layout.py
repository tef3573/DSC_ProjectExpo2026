"""
tests/test_web_layout.py
Sub-issue #278: Validate HTML5 structure, CSS responsive breakpoints, static assets,
and clean local HTTP server rendering for the web/ dashboard layout.
"""
from html.parser import HTMLParser
import http.server
import json
from pathlib import Path
import socketserver
import threading
import unittest
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"


class _DashboardHTMLParser(HTMLParser):
    """Strict HTML5 structural and attribute collector."""

    def __init__(self) -> None:
        super().__init__()
        self.tags = []
        self.ids = []
        self.stylesheets = []
        self.scripts = []

    def error(self, message):
        """Satisfy ParserBase abstract error hook."""

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attr_map = dict(attrs)
        if "id" in attr_map:
            self.ids.append(attr_map["id"])
        if tag == "link" and attr_map.get("rel") == "stylesheet" and "href" in attr_map:
            self.stylesheets.append(attr_map["href"])
        if tag == "script" and "src" in attr_map:
            self.scripts.append(attr_map["src"])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Silent static file handler serving web/ for local HTTP rendering checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        """Suppress HTTP access log noise during test runs."""


class TestWebDashboardLayout(unittest.TestCase):
    """Validation suite for #278: HTML5 structure, CSS rules, and local HTTP rendering."""

    def test_html5_semantic_landmarks_and_unique_ids(self) -> None:
        """Verify web/index.html contains valid semantic landmarks and unique element IDs."""
        index_path = WEB_DIR / "index.html"
        self.assertTrue(index_path.exists(), "web/index.html must exist")

        html_text = index_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_text)

        parser = _DashboardHTMLParser()
        parser.feed(html_text)

        self.assertEqual(len(parser.ids), len(set(parser.ids)), "Duplicate HTML IDs detected")
        for landmark in ("header", "nav", "main", "section", "article", "table", "footer"):
            self.assertIn(landmark, parser.tags, f"Missing HTML5 semantic tag: <{landmark}>")

        for required_id in (
            "strategy-select",
            "regime-select",
            "chart-equity-curve",
            "chart-drawdown",
            "chart-constituent-count",
            "macro-regime-table",
            "scorecard-table",
        ):
            self.assertIn(required_id, parser.ids, f"Missing required mount ID: {required_id}")

        for rel_path in parser.stylesheets + parser.scripts:
            self.assertTrue((WEB_DIR / rel_path).exists(), f"Referenced asset missing: {rel_path}")

    def test_css_dark_mode_and_widescreen_breakpoints(self) -> None:
        """Verify web/css/style.css defines dark-mode tokens and responsive/widescreen queries."""
        css_text = (WEB_DIR / "css" / "style.css").read_text(encoding="utf-8")
        for token in (
            "--bg-canvas:",
            "--bg-surface:",
            ".kpi-grid",
            ".panel-grid--primary",
            "@media (min-width: 1920px)",
            "@media (max-width: 1200px)",
            "@media (max-width: 680px)",
        ):
            self.assertIn(token, css_text, f"Missing CSS layout rule or token: {token}")

    def test_local_http_server_renders_all_dashboard_assets(self) -> None:
        """Serve web/ over loopback HTTP and verify 200 OK responses for all assets."""
        with socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{port}"
                for route in (
                    "/index.html",
                    "/css/style.css",
                    "/js/charts.js",
                    "/js/dashboard.js",
                    "/vendor/plotly.min.js",
                    "/data/summary_metrics.json",
                ):
                    with urllib.request.urlopen(f"{base_url}{route}", timeout=5) as resp:
                        self.assertEqual(resp.status, 200, f"HTTP failure on {route}")
                        body = resp.read()
                        self.assertGreater(len(body), 0, f"Empty payload on {route}")
                        if route.endswith(".json"):
                            parsed = json.loads(body.decode("utf-8"))
                            self.assertIn("timeseries_garch", parsed)
            finally:
                httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
