"""
api/live-market.py
Vercel Python Serverless Function exposing GET /api/live-market
by reusing fetch_live_market_snapshot() from src/terminal_server.py.
"""
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.terminal_server import fetch_live_market_snapshot  # noqa: E402


class handler(BaseHTTPRequestHandler):  # pylint: disable=invalid-name
    """Vercel serverless HTTP entrypoint for /api/live-market."""

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Return live market quotes, Treasury yields, headlines, and intraday bars."""
        payload = fetch_live_market_snapshot()
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "s-maxage=20, stale-while-revalidate=40")
        self.end_headers()
        self.wfile.write(encoded)
