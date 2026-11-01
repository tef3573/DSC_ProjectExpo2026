"""
tests/test_universe.py
Unit tests for point-in-time universe masking, execution lag, and constituent limits.
Sub-issue #78: Verify active constituent count N_t never exceeds roster length.
"""
from pathlib import Path
import unittest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"