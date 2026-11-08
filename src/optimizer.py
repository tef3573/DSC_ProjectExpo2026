"""
src/optimizer.py
Covariance Matrix Estimation and Convex Quality Optimization.
- Sub-issue #149: Module architecture and lookback return alignment.
- Sub-issue #150: Ledoit-Wolf shrinkage covariance estimator.
- Sub-issue #151: Low-sample matrix conditioning and strict positive semi-definiteness.
"""
from typing import Optional, Tuple
import logging
from anyio import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRADING_DAYS_PER_YEAR = 252

# 1-year trailing covariance estimation window
DEFAULT_LOOKBACK_DAYS = 252    

# Minimum ~3 months required for statistical validity
MIN_OBSERVATIONS_REQUIRED = 63  

 # Ridge floor to guarantee strict positive definiteness
REGULARIZATION_EPSILON = 1e-4     

""" Estimator class for asset return covariance matrices with shrinkage regularization."""
