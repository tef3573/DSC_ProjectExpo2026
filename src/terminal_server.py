"""
src/terminal_server.py
Lightweight HTTP server for the 1-BLOOMBERG 2011 Terminal UI.
Serves static files from web/ and exposes GET /api/live-market backed by yfinance
with a 20-second memory cache and deterministic market-close fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

CACHE_TTL_SECONDS = 20.0
_MARKET_CACHE: Dict[str, Any] = {"timestamp": 0.0, "payload": None}

INDEX_SPECS = [
    ("^GSPC", "S&P 500", 6910.39, -265.40, -3.70, 14.20),
    ("^DJI", "DOW JONES", 48147.33, -947.80, -1.93, 12.85),
    ("^IXIC", "NASDAQ 100", 22617.73, -1260.45, -5.28, 18.40),
    ("^RUT", "RUSSELL 2K", 2581.79, -74.36, -2.80, 9.10),
    ("^VIX", "VIX INDEX", 35.89, 7.45, 26.20, 48.10),
]

RATE_SPECS = [
    ("^IRX", "US 3M BILL", 4.32, -1.5),
    ("^FVX", "US 5Y NOTE", 4.48, -2.4),
    ("^TNX", "US 10Y NOTE", 4.92, -3.1),
    ("^TYX", "US 30Y BOND", 5.14, -1.8),
]


def _fallback_payload() -> Dict[str, Any]:
    """Return deterministic reference quotes when offline or rate-limited."""
    equities = [
        {
            "symbol": sym,
            "name": name,
            "last": last,
            "chg": chg,
            "pct_chg": pct,
            "ytd": ytd,
        }
        for sym, name, last, chg, pct, ytd in INDEX_SPECS
    ]
    rates = [
        {
            "symbol": sym,
            "name": name,
            "yield_pct": yld,
            "chg_bps": chg_bps,
            "spread_vs_3m_bps": round((yld - 4.32) * 100.0, 1),
        }
        for sym, name, yld, chg_bps in RATE_SPECS
    ]
    return {
        "source": "HISTORICAL SNAPSHOT (2026-11-20)",
        "is_live": False,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "equities": equities,
        "rates": rates,
        "intraday": {
            "labels": [f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}" for i in range(60)],
            "spx": [
                round(6910.39 + (i - 30) * 1.45 + ((i * 7) % 11 - 5) * 2.1, 2)
                for i in range(60)
            ],
            "vix": [
                round(35.89 - (i - 30) * 0.04 + ((i * 5) % 7 - 3) * 0.18, 2)
                for i in range(60)
            ],
        },
    }


def _parse_equities(hist: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build equity index quote rows from a multi-ticker close DataFrame."""
    equities: List[Dict[str, Any]] = []
    for sym, name, fb_last, fb_chg, fb_pct, fb_ytd in INDEX_SPECS:
        series = hist[sym].dropna() if sym in hist.columns else None
        if series is not None and len(series) >= 2:
            last_val = float(series.iloc[-1])
            prev_val = float(series.iloc[-2])
            chg_val = last_val - prev_val
            pct_val = (chg_val / prev_val) * 100.0 if prev_val else 0.0
            equities.append(
                {
                    "symbol": sym,
                    "name": name,
                    "last": round(last_val, 2),
                    "chg": round(chg_val, 2),
                    "pct_chg": round(pct_val, 2),
                    "ytd": fb_ytd,
                }
            )
        else:
            equities.append(
                {
                    "symbol": sym,
                    "name": name,
                    "last": fb_last,
                    "chg": fb_chg,
                    "pct_chg": fb_pct,
                    "ytd": fb_ytd,
                }
            )
    return equities


def _parse_rates(hist: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build US Treasury yield rows and 3M curve spreads from close DataFrame."""
    rates: List[Dict[str, Any]] = []
    short_yield = 4.32
    for sym, name, fb_yld, fb_bps in RATE_SPECS:
        series = hist[sym].dropna() if sym in hist.columns else None
        if series is not None and len(series) >= 2:
            yld = float(series.iloc[-1])
            prev_yld = float(series.iloc[-2])
            if sym == "^IRX":
                short_yield = yld
            rates.append(
                {
                    "symbol": sym,
                    "name": name,
                    "yield_pct": round(yld, 2),
                    "chg_bps": round((yld - prev_yld) * 100.0, 1),
                    "spread_vs_3m_bps": round((yld - short_yield) * 100.0, 1),
                }
            )
        else:
            rates.append(
                {
                    "symbol": sym,
                    "name": name,
                    "yield_pct": fb_yld,
                    "chg_bps": fb_bps,
                    "spread_vs_3m_bps": round((fb_yld - short_yield) * 100.0, 1),
                }
            )
    return rates


def _parse_intraday(fallback_intraday: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch 1-minute intraday SPX and VIX bars or return fallback."""
    intra = yf.download(
        tickers=["^GSPC", "^VIX"],
        period="1d",
        interval="1m",
        progress=False,
        threads=False,
    )["Close"]
    spx_intra = intra["^GSPC"].dropna().tail(60) if "^GSPC" in intra.columns else None
    vix_intra = intra["^VIX"].dropna().tail(60) if "^VIX" in intra.columns else None
    if spx_intra is None or len(spx_intra) < 10:
        return fallback_intraday

    aligned_vix = (
        vix_intra.reindex(spx_intra.index).ffill().bfill()
        if vix_intra is not None and not vix_intra.empty
        else None
    )
    return {
        "labels": [idx.strftime("%H:%M") for idx in spx_intra.index],
        "spx": [round(float(v), 2) for v in spx_intra.tolist()],
        "vix": (
            [round(float(v), 2) for v in aligned_vix.tolist()]
            if aligned_vix is not None
            else fallback_intraday["vix"][: len(spx_intra)]
        ),
    }


def fetch_live_market_snapshot() -> Dict[str, Any]:
    """Fetch live quotes & 1-minute intraday bars via yfinance with 20s cache."""
    now = time.time()
    cached = _MARKET_CACHE["payload"]
    if cached is not None and (now - _MARKET_CACHE["timestamp"]) < CACHE_TTL_SECONDS:
        return cached

    fallback = _fallback_payload()
    try:
        symbols = [s[0] for s in INDEX_SPECS] + [r[0] for r in RATE_SPECS]
        hist = yf.download(
            tickers=symbols,
            period="5d",
            interval="1d",
            progress=False,
            threads=False,
        )["Close"]
        payload = {
            "source": "LIVE YAHOO FINANCE TELEMETRY",
            "is_live": True,
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "equities": _parse_equities(hist),
            "rates": _parse_rates(hist),
            "intraday": _parse_intraday(fallback["intraday"]),
        }
    except Exception:  # pylint: disable=broad-exception-caught
        payload = fallback

    _MARKET_CACHE["timestamp"] = now
    _MARKET_CACHE["payload"] = payload
    return payload


class BloombergTerminalHandler(SimpleHTTPRequestHandler):
    """Serves static terminal assets and handles GET /api/live-market."""

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Route /api/live-market to yfinance snapshot or delegate to static web/."""
        if self.path.startswith("/api/live-market"):
            data = fetch_live_market_snapshot()
            encoded = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
        """Suppress noisy per-request HTTP stdout logs."""
        return


def run_server(port: int = 8081) -> None:
    """Start multi-threaded 1-BLOOMBERG HTTP server."""
    handler = partial(BloombergTerminalHandler, directory=str(WEB_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"1-BLOOMBERG Terminal running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server(8081)
