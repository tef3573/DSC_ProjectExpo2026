"""
api/live-market.py
Vercel Python WSGI entrypoint serving GET /api/live-market only.
Static assets are served natively by Vercel CDN from public/.
"""
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.terminal_server import fetch_live_market_snapshot  # noqa: E402


def app(environ, start_response):
    """Standard PEP 3333 WSGI application for /api/live-market."""
    payload = fetch_live_market_snapshot()
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "s-maxage=20, stale-while-revalidate=40"),
    ]
    start_response("200 OK", headers)
    return [body]
