"""
src/optimizer.py
Covariance Matrix Estimation and Convex Quality Optimization.
- Sub-issue #149: Module architecture and lookback return alignment.
- Sub-issue #150: Ledoit-Wolf shrinkage covariance estimator.
- Sub-issue #151: Low-sample matrix conditioning and strict positive semi-definiteness.
"""
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


