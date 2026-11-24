"""
src/terminal_server.py
Lightweight HTTP server for the 1-BLOOMBERG 2011 Terminal UI.
Serves static files from web/ and exposes GET /api/live-market backed by yfinance
and Investopedia/Yahoo financial RSS headlines with a 20-second memory cache.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import urllib.request
import xml.etree.ElementTree as ET

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

INVESTOPEDIA_RSS_URL = (
    "https://feeds-api.dotdashmeredith.com/v1/rss/google/"
    "f6a0e92b-be8d-4abb-9106-703b04059e19"
)


def _fetch_live_headlines() -> List[Dict[str, str]]:
    """Pull live financial headlines from Investopedia RSS and Yahoo Finance."""
    items: List[Dict[str, str]] = []
    try:
        req = urllib.request.Request(
            INVESTOPEDIA_RSS_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        for entry in root.findall(".//item")[:5]:
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            if title:
                items.append(
                    {
                        "time": datetime.now().strftime("%H:%M"),
                        "tag": "INVESTOPEDIA",
                        "title": title.upper(),
                        "url": link,
                    }
                )
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    try:
        yf_news = yf.Ticker("^GSPC").news or []
        for raw in yf_news[:6]:
            content = raw.get("content") if isinstance(raw, dict) else {}
            title = (
                (content.get("title") if isinstance(content, dict) else None)
                or raw.get("title")
                or ""
            ).strip()
            url = (
                (content.get("canonicalUrl", {}).get("url") if isinstance(content, dict) else None)
                or raw.get("link")
                or ""
            )
            if title and len(items) < 8:
                items.append(
                    {
                        "time": datetime.now().strftime("%H:%M"),
                        "tag": "MARKET WIRE",
                        "title": title.upper(),
                        "url": url,
                    }
                )
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    if not items:
        items = [
            {
                "time": "16:45",
                "tag": "INVESTOPEDIA",
                "title": "TREASURY YIELDS STABILIZE AS INSTITUTIONAL DESKS REPRICE FED PATH",
                "url": "https://www.investopedia.com/markets-news-4427704",
            },
            {
                "time": "16:20",
                "tag": "INVESTOPEDIA",
                "title": "FORTUNE 500 & RUSSELL 1000 CEO TURNOVER METRICS HIT MULTI-DECADE HIGH",
                "url": "https://www.investopedia.com/company-news-4427705",
            },
            {
                "time": "15:55",
                "tag": "MARKET WIRE",
                "title": "AI INFRASTRUCTURE & HEALTHCARE OUTPERFORM AS VOLATILITY CLUSTERS",
                "url": "https://www.investopedia.com/trading-news-4689736",
            },
            {
                "time": "15:30",
                "tag": "QUANT DESK",
                "title": "SURVIVORSHIP-BIAS-FREE POINT-IN-TIME INDEXING ELIMINATES LOOK-AHEAD DRIFT",
                "url": "",
            },
        ]
    return items


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
        "source": "HISTORICAL SNAPSHOT",
        "is_live": False,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "equities": equities,
        "rates": rates,
        "headlines": _fetch_live_headlines(),
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
    """Fetch live quotes, headlines & intraday bars via yfinance/RSS with 20s cache."""
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
            "source": "LIVE YAHOO FINANCE & INVESTOPEDIA WIRE",
            "is_live": True,
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "equities": _parse_equities(hist),
            "rates": _parse_rates(hist),
            "headlines": _fetch_live_headlines(),
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
