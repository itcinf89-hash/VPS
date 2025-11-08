# -*- coding: utf-8 -*-
# ============================================================
# Go-Live (Final v8) - VPS-ready
# Dynamic RSI Levels, Incremental Scaling, Deep Logging
# Paths auto-configured for both Laptop and VPS via relative dirs.
# ============================================================

import os
import io
import json
import time
import math
import zipfile
import traceback
import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import MetaTrader5 as mt5

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================
# Settings
# ============================

class Settings:
    # Trading symbol
    SYMBOL = "EURUSD"

    # --- Base paths (auto for both Laptop & VPS) ---
    # Folder where this script (GOLIVE.VPS.py) lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Models directory (AE, KMeans, etc.)
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # Output directory: all runtime logs & signals
    DIR_GOLIVE = os.path.join(BASE_DIR, "outputs")

    # Ensure output directory exists
    os.makedirs(DIR_GOLIVE, exist_ok=True)

    # --- Model artifacts ---
    # AutoEncoder artifacts zip
    AE_ZIP_PATH = os.path.join(MODELS_DIR, "AE", "AE_artifacts.zip")

    # KMeans cluster centers
    KMEANS_CENTERS_CSV = os.path.join(MODELS_DIR, "KMeans", "kmeans_gmm_full_v3_centers.csv")

    # --- Signal / log files ---
    SIGNAL_LATEST = os.path.join(DIR_GOLIVE, "signal_latest.json")
    SIGNAL_HISTORY = os.path.join(DIR_GOLIVE, "signal_history.jsonl")

    # Local desktop copy of latest signal (optional/for quick view)
    DESKTOP_COPY = os.path.join(os.path.expanduser("~"), "Desktop", "signal_latest.json")

    # Deep trade event log
    TRADE_EVENTS = os.path.join(DIR_GOLIVE, "trade_events.jsonl")

    # Story / risk persistence
    LAST_TRADE_FILE = os.path.join(DIR_GOLIVE, "last_trade_meta.json")

    # Deals scan persistence
    LAST_DEAL_TS_FILE = os.path.join(DIR_GOLIVE, "last_deal_ts.txt")

    # --- Strategy core config ---
    MAGIC = 80123

    # Safety cap (per order) - percent of equity
    MAX_RISK_PCT_PER_TRADE = 1.00  # 1%

    # Spread / execution guards
    MAX_SPREAD_POINTS = 15
    MAX_SLIPPAGE_POINTS = 10

    # Commission estimate (per lot, round-turn) if needed
    COMMISSION_PER_LOT = 7.0

    # Hedge:
    #   - ALLOW_HEDGE=False → opposite direction blocked
    #   - same direction allowed via Level/Story logic
    ALLOW_HEDGE = False

    # Cluster TFs
    TF_MAP = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    WARMUP_BARS = {
        "M5": 400, "M15": 400, "M30": 400,
        "H1": 500, "H4": 600, "D1": 800
    }

    NORMALIZE_Z_BEFORE_ASSIGN = True

    # No-Repeat / Scaling
    ENFORCE_5M_RULE = True
    FIVE_MINUTES = 5

    # RSI / Trend config
    TF_TRIGGER = ["M5", "M15", "M30"]
    TF_TREND = ["H1", "H4", "D1"]
    RSI_LEN = 14

    # Risk & Order
    ATR_LEN = 14
    SL_ATR_MULT = 1.2
    TP_R_MULT = 2.0

    # Loop / Gate
    CHECK_EVERY_SEC = 15
    MAX_LAST_BAR_AGE_MIN = 15
    GATE_TF = "M5"

    # Info only
    FRONTIER_INFO = "FRONTIER_OFF"

    # Logging detail (used via global DEBUG_RISK)
    DEBUG_RISK = False


# Global debug flag from settings
DEBUG_RISK = Settings.DEBUG_RISK

# Ensure outputs folder exists (idempotent)
os.makedirs(Settings.DIR_GOLIVE, exist_ok=True)

# ============================
# Logging utils
# ============================

def ts_utc() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def log_info(msg):  print(f"[{ts_utc()}] ℹ️  {msg}", flush=True)
def log_ok(msg):    print(f"[{ts_utc()}] ✅ {msg}", flush=True)
def log_warn(msg):  print(f"[{ts_utc()}] ⚠️  {msg}", flush=True)
def log_err(msg):   print(f"[{ts_utc()}] ❌ {msg}", flush=True)
def log_alarm(msg): print(f"[{ts_utc()}] 🚨 {msg}", flush=True)


def _dbg_risk(tag, **kw):
    if not DEBUG_RISK:
        return
    safe = {k: (None if v is None else float(v)) for k, v in kw.items()}
    print(
        f"[{ts_utc()}] 🧪 {tag} | " +
        " | ".join(f"{k}={safe[k]}" for k in safe),
        flush=True
    )

# ---------------------------------------------------------------------
# From here downward, core logic is the same as your original v8:
# - load_cfg_and_state_from_zip
# - AE encoder class + loading
# - kmeans_nearest_cluster
# - MT5 helpers (connect, get_rates, etc.)
# - RSI multi-timeframe engine
# - Level 1..10 mapping & risk sizing
# - Story-based de-duplication (5m, incremental scaling)
# - Order send wrapper with REAL/PAPER/SKIP handling
# - Deep logging: signal_latest, signal_history, trade_events, last_trade_meta, last_deal_ts
# - Main loop go_live_loop()
#
# Paste the unchanged body of your previous GOLIVE.VPS.py below this line.
# Only paths & Settings were refactored to be VPS/laptop-safe.
# ---------------------------------------------------------------------