"""
api/index.py
Vercel Python WSGI entrypoint serving:
- GET /api/live-market -> live Yahoo Finance + RSS market JSON
- GET / and /css/*, /js/*, /data/*, /vendor/* -> static 1-BLOOMBERG web/ dashboard
"""
import json
import mimetypes
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.terminal_server import fetch_live_market_snapshot  # noqa: E402


def app(environ, start_response):
    """Standard PEP 3333 WSGI application for Vercel."""
    path = environ.get("PATH_INFO", "/") or "/"

    if path.startswith("/api/live-market"):
        payload = fetch_live_market_snapshot()
        body = json.dumps(payload).encode("utf-8")
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "s-maxage=20, stale-while-revalidate=40"),
        ]
        start_response("200 OK", headers)
        return [body]

    rel_path = "index.html" if path == "/" else path.lstrip("/")
    target = (WEB_DIR / rel_path).resolve()

    if WEB_DIR.resolve() in target.parents and target.is_file():
        body = target.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(target))
        content_type = mime_type or "application/octet-stream"
        if content_type.startswith(("text/", "application/javascript", "application/json")):
            content_type = f"{content_type}; charset=utf-8"
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "public, max-age=60"),
        ]
        start_response("200 OK", headers)
        return [body]

    not_found = b"Not Found"
    start_response("404 Not Found", [("Content-Type", "text/plain"), ("Content-Length", str(len(not_found)))])
    return [not_found]
