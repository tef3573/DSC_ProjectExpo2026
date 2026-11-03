"""
src/backtest.py
Base Vectorized Portfolio Backtesting Engine.
- Issue #87 / Sub-issue #94: Daily 1/Nt asset return dot product.
- Issue #87 / Sub-issue #95: Dynamic cash drag allocation when Nt == 0.
- Issue #87 / Sub-issue #96: Division-by-zero protections during holidays and inactive periods.
"""

from pathlib import Path
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"
OUTPUT_RETURNS_PATH = DATA_DIR / "portfolio_daily_returns.csv"