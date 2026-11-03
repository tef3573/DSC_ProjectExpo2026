"""
tests/test_backtest.py
Unit tests verifying return calculation, equal-weighting, and single-asset boundary cases.
- Sub-issue #98: Test portfolio gross returns against single-asset boundary cases.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.backtest import compute_equal_weights, compute_asset_returns, run_base_backtest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"
