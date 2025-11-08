# -*- coding: utf-8 -*-
# ============================================================
# Go-Live (Final v8) - VPS-ready
# Dynamic RSI Levels, Incremental Scaling, Deep Logging:
# - Signal context (RSI/cluster/MT5 snapshot)
# - Trade events (REAL/PAPER/SKIP + reasons)
# - Deals (history_deals_get, MAGIC-filtered, PnL)
#
# Core:
#   * OLD cluster pipeline: context / story-id only
#   * Level 1..10: pure RSI engine, pick highest valid
#   * Risk: L1=0.1% ... L10=1.0% (capped by MAX_RISK_PCT_PER_TRADE)
#   * Same-story add: only if new Level > prev Level (incremental risk)
#   * No-repeat window: 5m story-based
#   * Hedge: opposite blocked if ALLOW_HEDGE=False
#
# VPS/Laptop ready:
#   * BASE_DIR = folder of this script
#   * MODELS_DIR = BASE_DIR/models
#       - AE/model files:    models/AE/AE_artifacts.zip
#       - KMeans centers:    models/KMeans/kmeans_gmm_full_v3_centers.csv
#   * Outputs: BASE_DIR/outputs
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
    # Symbol
    SYMBOL = "EURUSD"

    # --- Base paths (portable: laptop + VPS) ---
    # Folder where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Models directory (AE, KMeans, etc.)
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # Output directory: all runtime logs & signals
    DIR_GOLIVE = os.path.join(BASE_DIR, "outputs")
    os.makedirs(DIR_GOLIVE, exist_ok=True)

    # --- Model artifacts ---
    # AE artifacts zip
    AE_ZIP_PATH = os.path.join(MODELS_DIR, "AE", "AE_artifacts.zip")

    # KMeans cluster centers
    KMEANS_CENTERS_CSV = os.path.join(
        MODELS_DIR, "KMeans", "kmeans_gmm_full_v3_centers.csv"
    )

    # --- Signal / log files ---
    SIGNAL_LATEST = os.path.join(DIR_GOLIVE, "signal_latest.json")
    SIGNAL_HISTORY = os.path.join(DIR_GOLIVE, "signal_history.jsonl")

    # Local desktop copy (optional)
    DESKTOP_COPY = os.path.join(
        os.path.expanduser("~"), "Desktop", "signal_latest.json"
    )

    # Story meta (for incremental logic)
    LAST_TRADE_FILE = os.path.join(DIR_GOLIVE, "last_trade_meta.json")

    # Deep trade logs
    TRADE_EVENTS = os.path.join(DIR_GOLIVE, "trade_events.jsonl")

    # Deals logging
    LAST_DEAL_TS_FILE = os.path.join(DIR_GOLIVE, "last_deal_ts.txt")

    # --- Loop / Gate ---
    CHECK_EVERY_SEC = 15
    MAX_LAST_BAR_AGE_MIN = 15
    GATE_TF = "M5"  # gate on new closed M5 candle

    # Info only
    FRONTIER_INFO = "FRONTIER_OFF"

    # --- No-Repeat / Scaling ---
    ENFORCE_5M_RULE = True
    FIVE_MINUTES = 5  # story age

    # --- RSI / Trend config ---
    TF_TRIGGER = ["M5", "M15", "M30"]
    TF_TREND = ["H1", "H4", "D1"]
    RSI_LEN = 14

    # --- Risk & Order ---
    ATR_LEN = 14
    SL_ATR_MULT = 1.2
    TP_R_MULT = 2.0

    ENFORCE_SPREAD_LOCK = False
    MAX_SPREAD_POINTS = 25
    SLIPPAGE_ATR_FRACTION = 0.05
    COMMISSION_PER_LOT_ROUND_TURN = 0.0
    DEVIATION = 10
    MAGIC = 26031

    # Hedge:
    #   - ALLOW_HEDGE=False → opposite direction blocked
    #   - same direction: allowed via Level/Story logic
    ALLOW_HEDGE = False

    # Safety cap (per order)
    MAX_RISK_PCT_PER_TRADE = 0.01  # 1%

    # --- Cluster TFs ---
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

    # Logging detail for risk debug
    DEBUG_RISK = True


# Global debug flag
DEBUG_RISK = Settings.DEBUG_RISK

# Ensure outputs dir exists (idempotent)
os.makedirs(Settings.DIR_GOLIVE, exist_ok=True)

# ============================
# Logging utils
# ============================

def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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

# ============================
# Persistence helpers
# ============================

def save_signal_log(signal_log: Dict[str, Any]):
    try:
        # last signal
        with open(Settings.SIGNAL_LATEST, "w", encoding="utf-8") as f:
            json.dump(signal_log, f, ensure_ascii=False, indent=2)
        # local desktop copy
        try:
            with open(Settings.DESKTOP_COPY, "w", encoding="utf-8") as f:
                json.dump(signal_log, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # history (append)
        with open(Settings.SIGNAL_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal_log, ensure_ascii=False) + "\n")
    except Exception as e:
        log_warn(f"file logging failed: {e}")


def save_trade_event(event: Dict[str, Any]):
    """
    Deep trade journal:
      - Each order_send attempt (REAL / FAIL / PAPER)
      - Important SKIPs with reason
      - DEALs from history_deals_get (PnL)
    """
    try:
        event["_ts_utc"] = ts_utc()
        with open(Settings.TRADE_EVENTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        log_warn(f"trade_events logging failed: {e}")


def _load_last_trade_meta() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(Settings.LAST_TRADE_FILE):
            with open(Settings.LAST_TRADE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log_warn(f"read last_trade_meta failed: {e}")
    return None


def _save_last_trade_meta(meta: Dict[str, Any]):
    try:
        with open(Settings.LAST_TRADE_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_warn(f"save last_trade_meta failed: {e}")

# ============================
# MT5 helpers
# ============================

def _fetch_rates(symbol: str, timeframe, count=200) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    return pd.DataFrame(rates)


def _market_fresh_enough(tf_for_check) -> Tuple[bool, Optional[int]]:
    df = _fetch_rates(Settings.SYMBOL, tf_for_check, count=2)
    if df is None or df.empty:
        return False, None
    last_ts = int(df.iloc[-1]["time"])
    age_min = int((time.time() - last_ts) / 60)
    return age_min <= Settings.MAX_LAST_BAR_AGE_MIN, age_min


def _sym_props(symbol: str) -> Optional[Dict[str, Any]]:
    si = mt5.symbol_info(symbol)
    if not si:
        return None
    stop_level = getattr(si, "trade_stops_level", None)
    if stop_level is None:
        stop_level = getattr(si, "stops_level", None)
    return dict(
        point=si.point,
        digits=si.digits,
        tick_value=getattr(si, "trade_tick_value", None),
        tick_size=getattr(si, "trade_tick_size", None),
        vol_min=si.volume_min,
        vol_max=si.volume_max,
        vol_step=si.volume_step,
        fill_mode=getattr(si, "trade_fill_mode", mt5.ORDER_FILLING_FOK),
        min_stop_points=int(stop_level) if stop_level is not None else 0
    )


def get_mt5_status(symbol: str) -> Dict[str, Any]:
    """
    Market / account / positions snapshot
    stored inside signal_history for context.
    """
    tick = mt5.symbol_info_tick(symbol)
    acc = mt5.account_info()
    poss = mt5.positions_get(symbol=symbol) or []
    sym = mt5.symbol_info(symbol)

    spread_points = None
    if tick and sym and sym.point:
        try:
            spread_points = (tick.ask - tick.bid) / sym.point
        except Exception:
            spread_points = None

    positions_detail = []
    for p in poss:
        try:
            positions_detail.append({
                "ticket": int(p.ticket),
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                "type": int(p.type),
                "magic": int(p.magic),
                "symbol": p.symbol,
                "volume": float(p.volume),
                "price_open": float(p.price_open),
                "sl": float(p.sl),
                "tp": float(p.tp),
                "profit": float(p.profit),
                "swap": float(p.swap),
                "comment": p.comment,
            })
        except Exception:
            continue

    pnl_total = sum(p["profit"] for p in positions_detail) if positions_detail else 0.0

    return {
        "bid": float(tick.bid) if tick else None,
        "ask": float(tick.ask) if tick else None,
        "spread_points": round(spread_points, 1) if spread_points is not None else None,
        "positions": len(positions_detail),
        "positions_detail": positions_detail,
        "pnl_total": round(pnl_total, 2),
        "equity": float(acc.equity) if acc else None,
        "balance": float(acc.balance) if acc else None,
        "margin": float(acc.margin) if acc else None,
        "free_margin": float(acc.margin_free) if acc else None,
        "point": float(sym.point) if sym else None,
        "digits": int(sym.digits) if sym else None,
        "vol_min": float(sym.volume_min) if sym else None,
        "vol_step": float(sym.volume_step) if sym else None,
        "vol_max": float(sym.volume_max) if sym else None,
        "tick_value": float(getattr(sym, "trade_tick_value", 0.0)) if sym else None,
        "tick_size": float(getattr(sym, "trade_tick_size", 0.0)) if sym else None,
    }


def _normalize_price(price: float, digits: int) -> float:
    factor = 10 ** digits
    return math.floor(price * factor + 0.5) / factor


def _round_volume(vol: float, step: float, vmin: float, vmax: float) -> float:
    if step <= 0:
        return max(vmin, min(vol, vmax))
    steps = math.floor((vol - vmin) / step + 1e-9)
    v = vmin + steps * step
    return max(vmin, min(round(v, 3), vmax))

# ============================
# Indicators
# ============================

def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def _atr(h, l, c, n=14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1
    ).max(axis=1)
    return _rma(tr, n)


def _adx(h, l, c, n=14) -> pd.Series:
    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn.clip(lower=0)
    atr = _atr(h, l, c, n)
    plus_di = 100 * _rma(plus_dm, n) / atr.replace(0, np.nan)
    minus_di = 100 * _rma(minus_dm, n) / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() /
          (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return _rma(dx, n)


def minmax01(s: pd.Series) -> pd.Series:
    smin, smax = s.min(), s.max()
    if smax - smin == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - smin) / (smax - smin)).clip(0, 1)


def compute_kpis(df: pd.DataFrame,
                 n_atr=14, n_ma=50) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    adx = _adx(h, l, c, n_atr)
    atr = _atr(h, l, c, n_atr)
    atr_ma = atr.rolling(n_ma).mean()
    atr_rel = atr / atr_ma.replace(0, np.nan)
    adx_n = minmax01(adx)
    atr_rel_n = minmax01(atr_rel)
    clean = adx_n * (1 - (atr_rel_n - 0.5).abs())
    return pd.DataFrame(
        {"ADX": adx_n, "ATRrel": atr_rel_n, "Clean": minmax01(clean)},
        index=df.index
    )


def rsi_wilder(close: pd.Series, length=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ============================
# Cluster module (AE → z → KMeans)
# ============================

DEVICE = torch.device("cpu")


class EncoderOnly(nn.Module):
    def __init__(self, d_in=18, d_hidden=32, k=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, k)
        )

    def forward(self, x):
        return self.net(x)


def _zip_read_text(zf: zipfile.ZipFile, name: str) -> str:
    if name not in zf.namelist():
        raise FileNotFoundError(f"{name} not found in ZIP")
    return zf.read(name).decode("utf-8")


def load_cfg_and_state_from_zip(zip_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"AE zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        cfg_text = None
        for name in ("ae_inference_config.json", "ae_config.json"):
            if name in zf.namelist():
                cfg_text = _zip_read_text(zf, name)
                break
        if not cfg_text:
            raise KeyError("ae_inference_config.json/ae_config.json not found in ZIP.")
        cfg = json.loads(cfg_text)
        if "feature_columns" not in cfg or "input_dim" not in cfg:
            raise KeyError("config must include feature_columns and input_dim.")
        feat_cols_bt = cfg["feature_columns"]
        input_dim_cfg = int(cfg["input_dim"])

        # weights
        if "ae_encoder.pt" in zf.namelist():
            raw_state = torch.load(
                io.BytesIO(zf.read("ae_encoder.pt")),
                map_location="cpu"
            )
        elif "ae_full.pt" in zf.namelist() or "model.pt" in zf.namelist():
            pick = "ae_full.pt" if "ae_full.pt" in zf.namelist() else "model.pt"
            full = torch.load(io.BytesIO(zf.read(pick)), map_location="cpu")
            enc_like = {
                k[len("encoder."):]: v
                for k, v in full.items() if k.startswith("encoder.")
            }
            if enc_like:
                raw_state = enc_like
            else:
                raw_state = {
                    k[len("net."):]: v
                    for k, v in full.items() if k.startswith("net.")
                }
            if not raw_state:
                raise RuntimeError("encoder.* or net.* not found in state_dict.")
        else:
            raise FileNotFoundError("ae_encoder.pt / ae_full.pt / model.pt not found in ZIP.")

    if not isinstance(raw_state, dict):
        raise RuntimeError("Invalid state_dict")

    remapped = {}
    for k, v in raw_state.items():
        if k.startswith("net."):
            remapped[k] = v
        elif k.startswith("encoder.net."):
            remapped[k.replace("encoder.", "")] = v
        elif k.startswith("encoder."):
            remapped["net." + k.split(".", 1)[1]] = v
        elif k.split(".", 1)[0].isdigit():
            remapped["net." + k] = v
        else:
            remapped["net." + k] = v

    for need_k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"):
        if need_k not in remapped:
            raise RuntimeError(f"Missing weight: {need_k}")

    w0 = remapped["net.0.weight"]
    w2 = remapped["net.2.weight"]
    hidden, d_in_w = w0.shape
    k_latent, hidden2 = w2.shape
    if hidden != hidden2:
        raise RuntimeError("Hidden size mismatch.")
    if d_in_w != input_dim_cfg or d_in_w != len(feat_cols_bt):
        raise ValueError("AE input_dim mismatch.")

    enc = EncoderOnly(d_in=d_in_w, d_hidden=hidden, k=k_latent).to(DEVICE)
    enc.load_state_dict(remapped, strict=False)
    enc.eval()
    return enc, feat_cols_bt, {"d_in": d_in_w, "hidden": hidden, "k": k_latent}


def normalize_tf_names(name: str) -> str:
    name = name.replace("_5min", "_min5").replace("_15min", "_min15").replace("_30min", "_min30")
    name = name.replace("_1h", "_h1").replace("_4h", "_h4").replace("_1d", "_d1")
    return name


def map_features_bt_to_live(feature_columns_bt: List[str]) -> List[str]:
    return [normalize_tf_names(c) for c in feature_columns_bt]


def _l2_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)

# ============================
# Rates & KPIs for cluster
# ============================

def get_rates(symbol, tf_code, n) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close"]]


def compute_all_kpis_until(symbol: str, t_last) -> Dict[str, Dict[str, float]]:
    all_kpis = {}
    for tf, tf_code in Settings.TF_MAP.items():
        df = get_rates(symbol, tf_code, Settings.WARMUP_BARS[tf])
        if df.empty or len(df) < 20:
            continue
        kpi_df = compute_kpis(df)
        kpi_df = kpi_df.loc[kpi_df.index <= t_last]
        if kpi_df.empty:
            continue
        all_kpis[tf] = kpi_df.iloc[-1].to_dict()
    return all_kpis


def build_old_features(all_kpis: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    feat = {}
    tf_key = {
        "M5": "min5", "M15": "min15", "M30": "min30",
        "H1": "h1", "H4": "h4", "D1": "d1"
    }
    for tf in ["M5", "M15", "M30", "H1", "H4", "D1"]:
        if tf not in all_kpis:
            continue
        suffix = tf_key[tf]
        feat[f"mkt_adx_{suffix}"] = all_kpis[tf]["ADX"]
        feat[f"mkt_atr_rel_{suffix}"] = all_kpis[tf]["ATRrel"]
        feat[f"mkt_liq_{suffix}"] = all_kpis[tf]["Clean"]
    return feat


def infer_cluster_old(symbol: str, t_last) -> Tuple[int, Dict[str, Any]]:
    enc, feat_cols_bt, _ = load_cfg_and_state_from_zip(Settings.AE_ZIP_PATH)
    feat_cols_live = map_features_bt_to_live(feat_cols_bt)
    all_kpis = compute_all_kpis_until(symbol, t_last)
    feat = build_old_features(all_kpis)
    missing = [c for c in feat_cols_live if c not in feat]
    if missing:
        raise KeyError("Feature mismatch:\n" + "\n".join(missing))
    X = np.array([[float(feat[c]) for c in feat_cols_live]], dtype=np.float32)
    with torch.no_grad():
        z_vec = enc(torch.from_numpy(X)).cpu().numpy().reshape(-1)

    if not os.path.exists(Settings.KMEANS_CENTERS_CSV):
        raise FileNotFoundError("KMeans centers not found.")
    cdf = pd.read_csv(Settings.KMEANS_CENTERS_CSV)
    zcols = [c for c in cdf.columns if c.lower().startswith("z")]
    if not zcols:
        raise ValueError("centers csv needs z* columns.")
    centers = cdf[zcols].to_numpy(dtype=np.float32, copy=False)
    if centers.shape[1] != len(z_vec):
        raise ValueError("latent dim mismatch.")

    if Settings.NORMALIZE_Z_BEFORE_ASSIGN:
        z_input = _l2_norm(z_vec)
        centers_use = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    else:
        z_input = z_vec
        centers_use = centers

    diffs = centers_use - z_input.reshape(1, -1)
    d2 = np.sum(diffs * diffs, axis=1)
    idx = int(np.argmin(d2))
    dist = float(math.sqrt(d2[idx]))

    if "cluster_id" in cdf.columns:
        cluster_id = int(cdf.iloc[idx]["cluster_id"])
    elif "cid" in cdf.columns:
        cluster_id = int(cdf.iloc[idx]["cid"])
    else:
        cluster_id = idx

    meta = {
        "distance": round(dist, 6),
        "latent_dim": int(len(z_vec)),
        "version": "cluster_old_trade_new",
        "timestamp_utc": str(t_last)
    }
    return cluster_id, meta

# ============================
# RSI / Level / Signal engine
# ============================

def fetch_close_prices(symbol: str, timeframe: int,
                       count: int = 100) -> pd.Series:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rates)
    return df["close"]


def compute_rsi_multi_tf(symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    tf_map = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    rsi_now, rsi_prev = {}, {}
    for tf, mtf in tf_map.items():
        prices = fetch_close_prices(symbol, mtf, Settings.RSI_LEN + 2)
        if prices.empty:
            continue
        rsi_series = rsi_wilder(prices, Settings.RSI_LEN)
        rsi_now[tf] = float(rsi_series.iloc[-1])
        rsi_prev[tf] = float(rsi_series.iloc[-2])
    return rsi_now, rsi_prev


def _rsi_levels(level: int) -> Dict[str, float]:
    table = {
        1: (55, 45), 2: (56, 44), 3: (58, 42), 4: (60, 40), 5: (62, 38),
        6: (63, 37), 7: (65, 35), 8: (70, 30), 9: (75, 25), 10: (80, 20)
    }
    b, s = table.get(max(1, min(10, level)), (60, 40))
    return dict(
        BULL_ZONE=b, BEAR_ZONE=s,
        BULL_EXT=min(b + 15, 80),
        BEAR_EXT=max(s - 15, 20),
        BULL_SUP=min(b + 20, 85),
        BEAR_SUP=max(s - 20, 15),
    )


def rsi_state(rsi_value: float, level: int) -> str:
    thr = _rsi_levels(level)
    if math.isnan(rsi_value):
        return "neutral"
    if rsi_value >= thr["BULL_SUP"]:
        return "bull_super"
    if rsi_value >= thr["BULL_EXT"]:
        return "bull_extreme"
    if rsi_value >= thr["BULL_ZONE"]:
        return "bull_zone"
    if rsi_value <= thr["BEAR_SUP"]:
        return "bear_super"
    if rsi_value <= thr["BEAR_EXT"]:
        return "bear_extreme"
    if rsi_value <= thr["BEAR_ZONE"]:
        return "bear_zone"
    return "neutral"


def decide_trend(large_states: Dict[str, str], level: int) -> Tuple[Optional[str], Dict[str, Any]]:
    bulls = [tf for tf, s in large_states.items() if s.startswith("bull")]
    bears = [tf for tf, s in large_states.items() if s.startswith("bear")]
    score = (
        sum(2 if "super" in s else 1 for s in large_states.values() if s.startswith("bull"))
        - sum(2 if "super" in s else 1 for s in large_states.values() if s.startswith("bear"))
    )
    detail = {
        "counts": {"bull": len(bulls), "bear": len(bears)},
        "score": score,
        "level": level,
        "notes": []
    }

    def ok_bull(n): return len(bulls) >= n and len(bears) == 0
    def ok_bear(n): return len(bears) >= n and len(bulls) == 0

    if level <= 1:
        if len(bulls) >= 1 and len(bears) < 3:
            return "Bull", detail
        if len(bears) >= 1 and len(bulls) < 3:
            return "Bear", detail
    elif level == 2:
        if len(bulls) >= 1 and not any("super" in s for s in bears):
            return "Bull", detail
        if len(bears) >= 1 and not any("super" in s for s in bulls):
            return "Bear", detail
    elif level in (3, 4):
        if ok_bull(2):
            return "Bull", detail
        if ok_bear(2):
            return "Bear", detail
    elif level == 5:
        if ok_bull(2) and any(
            x in s for tf, s in large_states.items()
            for x in ("extreme", "super") if s.startswith("bull")
        ):
            return "Bull", detail
        if ok_bear(2) and any(
            x in s for tf, s in large_states.items()
            for x in ("extreme", "super") if s.startswith("bear")
        ):
            return "Bear", detail
    elif level == 6:
        if ok_bull(3):
            return "Bull", detail
        if ok_bear(3):
            return "Bear", detail
    elif level == 7:
        if ok_bull(3) and any("super" in s for s in large_states.values() if s.startswith("bull")):
            return "Bull", detail
        if ok_bear(3) and any("super" in s for s in large_states.values() if s.startswith("bear")):
            return "Bear", detail
    elif level == 8:
        if all("super" in s for s in large_states.values() if s.startswith("bull")):
            return "Bull", detail
        if all("super" in s for s in large_states.values() if s.startswith("bear")):
            return "Bear", detail
    elif level == 9:
        if all("super" in s for s in large_states.values() if s.startswith("bull")) and score >= 5:
            return "Bull", detail
        if all("super" in s for s in large_states.values() if s.startswith("bear")) and score <= -5:
            return "Bear", detail
    else:
        if all("super" in s for s in large_states.values() if s.startswith("bull")) and score >= 6:
            return "Bull", detail
        if all("super" in s for s in large_states.values() if s.startswith("bear")) and score <= -6:
            return "Bear", detail

    detail["notes"].append("NoTrend: conditions not met")
    return None, detail


def detect_trigger(rsi_now: Dict[str, float],
                   rsi_prev: Dict[str, float],
                   trend: Optional[str],
                   level: int) -> Tuple[bool, List[str], int]:
    triggers = []
    for tf in rsi_now:
        if tf not in Settings.TF_TRIGGER:
            continue
        prev = rsi_prev.get(tf, math.nan)
        now = rsi_now.get(tf, math.nan)
        if math.isnan(prev) or math.isnan(now):
            continue

        if trend == "Bear":
            if prev > 50 and now < 50:
                triggers.append(f"{tf}:Cross50Down")
            elif prev > 70 and now < 70:
                triggers.append(f"{tf}:ExitOverbought")
            elif now < prev and now < 45:
                triggers.append(f"{tf}:BearSlope")
        elif trend == "Bull":
            if prev < 50 and now > 50:
                triggers.append(f"{tf}:Cross50Up")
            elif prev < 30 and now > 30:
                triggers.append(f"{tf}:ExitOversold")
            elif now > prev and now > 55:
                triggers.append(f"{tf}:BullSlope")

    n = len(triggers)
    if n == 0:
        strength = 0
    elif n == 1:
        strength = 1
    elif n == 2:
        strength = 2
    else:
        strength = 3

    ok = False
    if level <= 2 and strength >= 1:
        ok = True
    elif 3 <= level <= 5 and strength >= 2:
        ok = True
    elif 6 <= level <= 8 and strength >= 2 and n >= 2:
        ok = True
    elif level >= 9 and strength >= 3:
        ok = True

    return ok, triggers, strength


def decide_state(level: int,
                 trend: Optional[str],
                 trend_detail: Dict[str, Any],
                 trigger_strength: int) -> Tuple[str, Dict[str, Any]]:
    score = trend_detail.get("score", 0) if trend_detail else 0
    notes = []
    state = "Probe"

    if trend in ("Bull", "Bear") and trigger_strength >= 1:
        state = "Starter"
        notes.append("Starter")

    if trend in ("Bull", "Bear") and trigger_strength >= 2 and score >= (1 if level <= 5 else 2):
        state = "Standard"
        notes.append("Standard")

    if level >= 8 and trigger_strength >= 3 and score >= 4:
        state = "High-Conviction"
        notes.append("High-Conviction")

    return state, {"trend_score": score, "notes": notes}


def risk_pct_from_level(level: int) -> float:
    level = max(1, min(10, int(level)))
    return 0.001 * level


def compute_for_level(rsi_now: Dict[str, float],
                      rsi_prev: Dict[str, float],
                      level: int) -> Optional[Dict[str, Any]]:
    states = {tf: rsi_state(val, level) for tf, val in rsi_now.items()}
    large_states = {tf: states[tf] for tf in Settings.TF_TREND if tf in states}

    trend, trend_detail = decide_trend(large_states, level)
    trigger_ok, trigger_flags, trigger_strength = detect_trigger(
        rsi_now, rsi_prev, trend, level
    )
    state, state_meta = decide_state(level, trend, trend_detail, trigger_strength)

    if not trigger_ok:
        return None

    if trend is None:
        # Probe بدون ترند مشخص
        if state != "Probe":
            return None
        if rsi_now.get("M15", 50.0) >= 50.0 or rsi_now.get("M5", 50.0) >= 50.0:
            signal = "Buy"
            reason = f"L{level}:NoTrend-ProbeUp"
        else:
            signal = "Sell"
            reason = f"L{level}:NoTrend-ProbeDown"
    else:
        if trend == "Bull":
            signal = "Buy"
            reason = f"L{level}:BullTrend+{state}"
        else:
            signal = "Sell"
            reason = f"L{level}:BearTrend+{state}"

    target_risk_pct = min(risk_pct_from_level(level),
                          Settings.MAX_RISK_PCT_PER_TRADE)

    return {
        "level": int(level),
        "state": state,
        "state_meta": state_meta,
        "trend": trend or "None",
        "trend_detail": trend_detail,
        "rsi_now": rsi_now,
        "rsi_prev": rsi_prev,
        "trigger_flags": trigger_flags,
        "trigger_strength": int(trigger_strength),
        "signal": signal,
        "reason": reason,
        "risk": {"target_pct": float(target_risk_pct)},
    }


def compute_signal_multi_level(rsi_now: Dict[str, float],
                               rsi_prev: Dict[str, float]) -> Optional[Dict[str, Any]]:
    best = None
    for lvl in range(1, 11):
        cand = compute_for_level(rsi_now, rsi_prev, lvl)
        if cand is not None:
            best = cand  # highest valid
    return best

# ============================
# Risk & Execution helpers
# ============================

def _atr_tf(symbol: str, timeframe, length: int) -> Optional[float]:
    df = _fetch_rates(symbol, timeframe, count=length + 2)
    if df is None or df.empty:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean().iloc[-1]
    return float(atr)


def _effective_risk_inputs(signal: str,
                           price_entry: float,
                           sl_price: float,
                           point: float,
                           spread_points: Optional[float],
                           atr_points: float,
                           tick_value: Optional[float],
                           tick_size: Optional[float],
                           commission_round_turn: float,
                           lots: float) -> Tuple[float, float]:
    if signal == "Buy":
        base_points = max(0.0, (price_entry - sl_price) / point)
    else:
        base_points = max(0.0, (sl_price - price_entry) / point)

    slip_pts = Settings.SLIPPAGE_ATR_FRACTION * atr_points

    if tick_value and tick_size and point:
        loss_per_point_per_lot = tick_value * (point / tick_size)
    else:
        loss_per_point_per_lot = 0.0

    comm_pts = 0.0
    if loss_per_point_per_lot > 0 and lots > 0 and commission_round_turn > 0:
        comm_pts = commission_round_turn / (loss_per_point_per_lot * lots)

    adj_points = base_points + (spread_points or 0.0) + slip_pts + comm_pts
    return float(adj_points), float(loss_per_point_per_lot)


def _size_to_target(equity: float,
                    target_risk_pct: float,
                    adj_points: float,
                    loss_per_point_per_lot: float,
                    vol_min: float, vol_max: float, vol_step: float) -> float:
    if adj_points <= 0 or loss_per_point_per_lot <= 0:
        return vol_min
    raw = (equity * target_risk_pct) / (adj_points * loss_per_point_per_lot)
    return _round_volume(raw, vol_step, vol_min, vol_max)


def _ensure_min_stop(sl_price: float,
                     price_entry: float,
                     signal: str,
                     point: float,
                     digits: int,
                     min_stop_points: int) -> float:
    if signal == "Buy":
        base_pts = (price_entry - sl_price) / point
        if base_pts >= min_stop_points:
            return sl_price
        delta = min_stop_points - base_pts
        return _normalize_price(sl_price - delta * point, digits)
    else:
        base_pts = (sl_price - price_entry) / point
        if base_pts >= min_stop_points:
            return sl_price
        delta = min_stop_points - base_pts
        return _normalize_price(sl_price + delta * point, digits)


def _alarm_if_over_risk(eff_risk: Optional[float], cap: float):
    if eff_risk is not None and eff_risk > cap + 1e-12:
        pct = round(eff_risk * 100, 3)
        cap_pct = round(cap * 100, 3)
        log_alarm(f"effective_risk {pct}% > MAX_RISK {cap_pct}%")


def _send_with_fill_fallback(req_base: Dict[str, Any],
                             fill_candidates: List[int]):
    last_res = None
    used = None
    for f in fill_candidates:
        req = dict(req_base)
        req["type_filling"] = f
        res = mt5.order_send(req)
        last_res = res
        print(
            f"[{ts_utc()}] 📤 try_filling={f} "
            f"retcode={getattr(res, 'retcode', None)} "
            f"comment={getattr(res, 'comment', '')}",
            flush=True
        )
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            used = f
            break
    return last_res, used

# ============================
# Incremental risk helper
# ============================

def _within_5m(ts_iso: str) -> bool:
    if not Settings.ENFORCE_5M_RULE:
        return True
    try:
        s = ts_iso.replace("Z", "")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) < timedelta(minutes=Settings.FIVE_MINUTES)
    except Exception:
        return False


def _compute_incremental_risk_pct(symbol: str,
                                  cluster_key: str,
                                  state: str,
                                  level: int,
                                  direction: str) -> Tuple[float, bool]:
    """
    Story = (symbol, cluster_key, state, direction) در ۵ دقیقه اخیر.
    اگر همان Story و Level جدید > قبلی → فقط اختلاف ریسک.
    """
    base_risk = risk_pct_from_level(level)
    last_meta = _load_last_trade_meta()
    if not last_meta:
        return base_risk, False

    if Settings.ENFORCE_5M_RULE and not _within_5m(last_meta.get("timestamp_iso", "")):
        return base_risk, False

    same_symbol = (last_meta.get("symbol") == symbol)
    same_cluster = (last_meta.get("cluster_key") == cluster_key)
    same_state = (last_meta.get("state") == state)
    same_dir = (last_meta.get("direction") == direction)

    if same_symbol and same_cluster and same_state and same_dir:
        prev_level = int(last_meta.get("level", 0))
        prev_risk = risk_pct_from_level(prev_level)
        if level > prev_level:
            inc = max(0.0, base_risk - prev_risk)
            return inc, True
        else:
            return 0.0, True  # block non-improved
    return base_risk, False

# ============================
# No-Repeat duplicate guard
# ============================

def _dup_block(cluster_key: str,
               state: str,
               level: int,
               symbol: str,
               direction: str) -> Tuple[bool, Dict[str, Any]]:
    """
    اگر در ۵ دقیقه اخیر همان (symbol, cluster_key, state, direction)
    با Level >= level بوده → بلاک سیگنال تکراری.
    """
    last_meta = _load_last_trade_meta()
    if not last_meta:
        return False, {}

    if Settings.ENFORCE_5M_RULE and not _within_5m(last_meta.get("timestamp_iso", "")):
        return False, {}

    same_symbol = (last_meta.get("symbol") == symbol)
    same_cluster = (last_meta.get("cluster_key") == cluster_key)
    same_state = (last_meta.get("state") == state)
    same_dir = (last_meta.get("direction") == direction)
    last_level = int(last_meta.get("level", 1))

    if same_symbol and same_cluster and same_state and same_dir and level <= last_level:
        return True, {
            "exec_status": "SKIPPED_DUP_CLUSTER_STATE_LEVEL",
            "last_trade_at": last_meta.get("timestamp_iso"),
            "cluster_key": cluster_key,
            "state": state,
            "last_level": last_level,
            "new_level": level
        }
    return False, {}

# ============================
# Deals logging helpers
# ============================

def _load_last_deal_ts() -> datetime:
    try:
        if os.path.exists(Settings.LAST_DEAL_TS_FILE):
            with open(Settings.LAST_DEAL_TS_FILE, "r", encoding="utf-8") as f:
                s = f.read().strip()
                if s:
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
    except Exception:
        pass
    return datetime.now(timezone.utc) - timedelta(days=7)


def _save_last_deal_ts(ts: datetime):
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        with open(Settings.LAST_DEAL_TS_FILE, "w", encoding="utf-8") as f:
            f.write(ts.isoformat())
    except Exception as e:
        log_warn(f"save last_deal_ts failed: {e}")


def log_new_deals():
    last_ts = _load_last_deal_ts()
    now_utc = datetime.now(timezone.utc)

    deals = mt5.history_deals_get(last_ts, now_utc)
    if deals is None:
        return

    max_ts = last_ts
    for d in deals:
        try:
            if int(d.magic) != int(Settings.MAGIC):
                continue
            dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
            if dt <= last_ts:
                continue

            save_trade_event({
                "event": "DEAL",
                "timestamp": dt.isoformat(),
                "ticket": int(d.ticket),
                "order": int(d.order),
                "position_id": int(d.position_id),
                "symbol": d.symbol,
                "type": int(d.type),
                "entry": int(d.entry),
                "volume": float(d.volume),
                "price": float(d.price),
                "profit": float(d.profit),
                "swap": float(d.swap),
                "commission": float(d.commission),
                "magic": int(d.magic),
                "comment": d.comment,
            })

            if dt > max_ts:
                max_ts = dt
        except Exception:
            continue

    if max_ts > last_ts:
        _save_last_deal_ts(max_ts)

# ============================
# Execution
# ============================

def place_order_or_paper(signal_log: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    symbol = signal_log["symbol"]
    state = signal_log["state"]
    level = int(signal_log["level"])
    direction = signal_log["signal"]  # "Buy" / "Sell"
    cluster_key = signal_log.get("cluster_key", "")

    base_event = {
        "symbol": symbol,
        "direction": direction,
        "level": level,
        "state": state,
        "cluster_key": cluster_key,
        "magic": Settings.MAGIC,
    }

    # Opposite hedge guard
    poss = mt5.positions_get(symbol=symbol) or []
    if not Settings.ALLOW_HEDGE:
        opp_dir = any(
            (p.type == mt5.ORDER_TYPE_BUY and direction == "Sell") or
            (p.type == mt5.ORDER_TYPE_SELL and direction == "Buy")
            for p in poss
        )
        if opp_dir:
            ev = dict(base_event)
            ev["event"] = "SKIP_HEDGE_OPPOSITE"
            ev["reason"] = "Opposite position exists and hedge disabled"
            save_trade_event(ev)
            return "SKIP", {"exec_status": "SKIPPED_HEDGE_PROHIBITED_OPPOSITE"}

    # Incremental risk
    risk_pct, is_incremental = _compute_incremental_risk_pct(
        symbol=symbol,
        cluster_key=cluster_key,
        state=state,
        level=level,
        direction=direction
    )
    if risk_pct <= 0:
        ev = dict(base_event)
        ev["event"] = "SKIP_NO_INCREMENTAL_RISK"
        ev["reason"] = "No additional risk allowed for same story"
        save_trade_event(ev)
        return "SKIP", {"exec_status": "SKIPPED_NO_INCREMENTAL_RISK"}

    target_risk_pct = min(risk_pct, Settings.MAX_RISK_PCT_PER_TRADE)

    tick = mt5.symbol_info_tick(symbol)
    props = _sym_props(symbol)
    if not (tick and props):
        ev = dict(base_event)
        ev["event"] = "SKIP_NO_SYMBOL_OR_PROPS"
        save_trade_event(ev)
        return "SKIP", {"exec_status": "SKIPPED_NO_SYMBOL_OR_PROPS"}

    spread_points = None
    if props["point"]:
        try:
            spread_points = (tick.ask - tick.bid) / props["point"]
        except Exception:
            spread_points = None

    if (Settings.ENFORCE_SPREAD_LOCK and
        spread_points is not None and
        spread_points > Settings.MAX_SPREAD_POINTS):
        ev = dict(base_event)
        ev["event"] = "SKIP_SPREAD_LOCK"
        ev["spread_points"] = float(spread_points)
        save_trade_event(ev)
        return "SKIP", {
            "exec_status": "SKIPPED_SPREAD_LOCK",
            "spread_points": round(spread_points, 1)
        }

    price_entry = tick.ask if direction == "Buy" else tick.bid
    atr = _atr_tf(symbol, mt5.TIMEFRAME_H1, Settings.ATR_LEN)
    if atr is None or atr <= 0:
        ev = dict(base_event)
        ev["event"] = "SKIP_NO_ATR"
        save_trade_event(ev)
        return "SKIP", {"exec_status": "SKIPPED_NO_ATR"}

    digits = props["digits"]
    point = props["point"]

    sl_dist_price = Settings.SL_ATR_MULT * atr
    if direction == "Buy":
        sl_0 = _normalize_price(price_entry - sl_dist_price, digits)
        tp_0 = _normalize_price(price_entry + Settings.TP_R_MULT * (price_entry - sl_0), digits)
    else:
        sl_0 = _normalize_price(price_entry + sl_dist_price, digits)
        tp_0 = _normalize_price(price_entry - Settings.TP_R_MULT * (sl_0 - price_entry), digits)

    atr_points = (sl_dist_price / point) if point else 0.0

    adj_points_0, loss_per_pt_per_lot = _effective_risk_inputs(
        signal=direction,
        price_entry=price_entry,
        sl_price=sl_0,
        point=point,
        spread_points=spread_points,
        atr_points=atr_points,
        tick_value=props["tick_value"],
        tick_size=props["tick_size"],
        commission_round_turn=Settings.COMMISSION_PER_LOT_ROUND_TURN,
        lots=1.0
    )

    acc = mt5.account_info()
    if not acc:
        ev = dict(base_event)
        ev["event"] = "SKIP_NO_ACCOUNT"
        save_trade_event(ev)
        return "SKIP", {"exec_status": "SKIPPED_NO_ACCOUNT"}

    equity = float(acc.equity)
    vol_calc = _size_to_target(
        equity, target_risk_pct,
        adj_points_0, loss_per_pt_per_lot,
        props["vol_min"], props["vol_max"], props["vol_step"]
    )

    # Min stop distance
    min_stop_points = props["min_stop_points"]
    if point:
        base_pts = (price_entry - sl_0) / point if direction == "Buy" else (sl_0 - price_entry) / point
    else:
        base_pts = 0.0

    if base_pts < float(min_stop_points):
        sl_0 = _ensure_min_stop(sl_0, price_entry, direction, point, digits, min_stop_points)
        adj_points_0, loss_per_pt_per_lot = _effective_risk_inputs(
            signal=direction,
            price_entry=price_entry,
            sl_price=sl_0,
            point=point,
            spread_points=spread_points,
            atr_points=atr_points,
            tick_value=props["tick_value"],
            tick_size=props["tick_size"],
            commission_round_turn=Settings.COMMISSION_PER_LOT_ROUND_TURN,
            lots=1.0
        )

    effective_risk_pct = None
    if equity > 0 and loss_per_pt_per_lot > 0 and vol_calc > 0:
        effective_risk_pct = (adj_points_0 * loss_per_pt_per_lot * vol_calc) / equity

    _dbg_risk(
        "NORMAL",
        equity=equity,
        level=level,
        target_risk_pct=target_risk_pct,
        price_entry=price_entry,
        atr=atr,
        sl_points=(price_entry - sl_0) / point if direction == "Buy" else (sl_0 - price_entry) / point,
        min_stop_points=min_stop_points,
        adj_points=adj_points_0,
        loss_per_pt=loss_per_pt_per_lot,
        vol=vol_calc,
        eff_risk_pct=effective_risk_pct,
        incremental=is_incremental
    )
    _alarm_if_over_risk(effective_risk_pct, Settings.MAX_RISK_PCT_PER_TRADE)

    # REAL order path
    if (effective_risk_pct is not None and
        effective_risk_pct <= Settings.MAX_RISK_PCT_PER_TRADE + 1e-9 and
        vol_calc >= props["vol_min"]):

        comment = f"GoLive L{level} {direction}"
        if is_incremental:
            comment += " Add"

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "type": mt5.ORDER_TYPE_BUY if direction == "Buy" else mt5.ORDER_TYPE_SELL,
            "volume": float(vol_calc),
            "price": _normalize_price(price_entry, digits),
            "sl": float(sl_0),
            "tp": float(tp_0),
            "deviation": Settings.DEVIATION,
            "magic": Settings.MAGIC,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        fill_list = [
            props["fill_mode"],
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_FOK
        ]
        res, used_fill = _send_with_fill_fallback(req, fill_list)

        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            ev = dict(base_event)
            ev.update({
                "event": "ORDER_SEND_REAL_DONE",
                "volume": float(vol_calc),
                "price": req["price"],
                "sl": float(sl_0),
                "tp": float(tp_0),
                "effective_risk_pct": float(effective_risk_pct),
                "adj_points": float(adj_points_0),
                "spread_points": float(spread_points) if spread_points is not None else None,
                "retcode": int(res.retcode),
                "order": int(getattr(res, "order", 0)),
                "deal": int(getattr(res, "deal", 0)),
                "comment": res.comment,
                "incremental": bool(is_incremental),
                "fill_used": int(used_fill) if used_fill is not None else None,
            })
            save_trade_event(ev)

            return "REAL", {
                "exec_status": "DONE_ADD" if is_incremental else "DONE",
                "ticket": getattr(res, "order", None),
                "volume": float(vol_calc),
                "price": req["price"],
                "sl": float(sl_0),
                "tp": float(tp_0),
                "effective_risk_pct": float(effective_risk_pct),
                "adj_points": float(adj_points_0),
                "spread_points": round(spread_points, 1) if spread_points is not None else None,
                "fill_used": int(used_fill) if used_fill is not None else None,
                "incremental": bool(is_incremental)
            }

        # Send fail
        ev = dict(base_event)
        ev.update({
            "event": "ORDER_SEND_FAIL",
            "request": req,
            "retcode": int(getattr(res, "retcode", -1)) if res else -1,
            "comment": getattr(res, "comment", ""),
            "incremental": bool(is_incremental),
        })
        save_trade_event(ev)

        return "SKIP", {
            "exec_status": "SEND_FAIL",
            "retcode": int(getattr(res, "retcode", -1)) if res else -1,
            "retcode_comment": getattr(res, "comment", ""),
            "incremental": bool(is_incremental)
        }

    # PAPER fallback
    ev = dict(base_event)
    ev.update({
        "event": "PAPER_TRADE",
        "volume": float(props["vol_min"]),
        "price": float(price_entry),
        "sl": float(sl_0),
        "tp": float(tp_0),
        "effective_risk_pct": float(effective_risk_pct) if effective_risk_pct is not None else None,
        "adj_points": float(adj_points_0),
        "spread_points": float(spread_points) if spread_points is not None else None,
        "min_lot_mode": True,
        "incremental": bool(is_incremental),
        "max_risk_cap": float(Settings.MAX_RISK_PCT_PER_TRADE),
    })
    save_trade_event(ev)

    return "PAPER", {
        "exec_status": "PAPER",
        "volume": float(props["vol_min"]),
        "price": float(price_entry),
        "sl": float(sl_0),
        "tp": float(tp_0),
        "effective_risk_pct": float(effective_risk_pct) if effective_risk_pct is not None else None,
        "adj_points": float(adj_points_0),
        "spread_points": round(spread_points, 1) if spread_points is not None else None,
        "min_lot_mode": True,
        "incremental": bool(is_incremental),
        "max_risk_cap": float(Settings.MAX_RISK_PCT_PER_TRADE)
    }

# ============================
# Main loop (M5 gated)
# ============================

def go_live_loop():
    log_info("Connecting to MT5 ...")
    if not mt5.initialize():
        log_err("MT5 initialize failed.")
        return
    if not mt5.symbol_select(Settings.SYMBOL, True):
        log_err(f"Symbol not available: {Settings.SYMBOL}")
        mt5.shutdown()
        return
    log_ok(f"MT5 OK | Symbol: {Settings.SYMBOL}")

    last_m5_close = None

    try:
        while True:
            try:
                # 1) Log new DEALs
                log_new_deals()

                # 2) Freshness check
                fresh_ok, age_min = _market_fresh_enough(mt5.TIMEFRAME_M5)
                if not fresh_ok:
                    age_txt = f"{age_min}m" if age_min is not None else "NO_DATA"
                    partial = {
                        "timestamp": ts_utc(),
                        "symbol": Settings.SYMBOL,
                        "order": {
                            "exec_status": "SKIPPED_STALE_CANDLE",
                            "message": f"STALE_CANDLE age={age_txt}"
                        },
                        "guards": {
                            "frontier": Settings.FRONTIER_INFO,
                            "stale_candle": True,
                            "spread_lock": Settings.ENFORCE_SPREAD_LOCK,
                            "dup_block": False
                        }
                    }
                    save_signal_log(partial)
                    save_trade_event({
                        "event": "SKIP_STALE_CANDLE",
                        "symbol": Settings.SYMBOL,
                        "age_min": age_min,
                        "note": "Market data too old or missing"
                    })
                    log_warn(f"Stale candle: age={age_txt}")
                    time.sleep(Settings.CHECK_EVERY_SEC)
                    continue

                # 3) Gate on new closed M5 candle
                m5_df = get_rates(
                    Settings.SYMBOL,
                    Settings.TF_MAP[Settings.GATE_TF],
                    Settings.WARMUP_BARS["M5"]
                )
                if m5_df.empty:
                    log_warn("No M5 data.")
                    save_trade_event({
                        "event": "SKIP_NO_M5_DATA",
                        "symbol": Settings.SYMBOL
                    })
                    time.sleep(Settings.CHECK_EVERY_SEC)
                    continue

                t_last = m5_df.index[-1]
                if last_m5_close is not None and t_last == last_m5_close:
                    time.sleep(Settings.CHECK_EVERY_SEC)
                    continue
                last_m5_close = t_last

                # 4) Cluster (OLD pipeline)
                try:
                    cluster_id, cluster_meta = infer_cluster_old(Settings.SYMBOL, t_last)
                    cluster_key = f"{cluster_meta.get('version', 'v1')}-cid:{int(cluster_id)}"
                    log_ok(
                        f"Cluster OK → id={cluster_id} "
                        f"| key={cluster_key} "
                        f"| dist={cluster_meta.get('distance')}"
                    )
                except Exception as e:
                    msg = f"Cluster error: {e}"
                    partial = {
                        "timestamp": ts_utc(),
                        "symbol": Settings.SYMBOL,
                        "cluster_id": None,
                        "cluster_key": "",
                        "cluster_meta": {"error": str(e)},
                        "order": {
                            "exec_status": "SKIPPED_NO_CLUSTER",
                            "message": str(e)
                        },
                        "guards": {
                            "frontier": Settings.FRONTIER_INFO,
                            "stale_candle": False,
                            "spread_lock": Settings.ENFORCE_SPREAD_LOCK,
                            "dup_block": False
                        }
                    }
                    save_signal_log(partial)
                    save_trade_event({
                        "event": "SKIP_NO_CLUSTER",
                        "symbol": Settings.SYMBOL,
                        "error": str(e)
                    })
                    log_warn(msg)
                    time.sleep(Settings.CHECK_EVERY_SEC)
                    continue

                # 5) RSI multi-TF
                rsi_now, rsi_prev = compute_rsi_multi_tf(Settings.SYMBOL)
                if not rsi_now:
                    log_warn("RSI missing.")
                    save_trade_event({
                        "event": "SKIP_NO_RSI",
                        "symbol": Settings.SYMBOL
                    })
                    time.sleep(Settings.CHECK_EVERY_SEC)
                    continue

                # 6) Dynamic Level 1..10
                core = compute_signal_multi_level(rsi_now, rsi_prev)
                if core is None:
                    partial = {
                        "timestamp": ts_utc(),
                        "symbol": Settings.SYMBOL,
                        "cluster_id": int(cluster_id),
                        "cluster_key": cluster_key,
                        "cluster_meta": cluster_meta,
                        "order": {
                            "exec_status": "SKIPPED_NO_VALID_LEVEL",
                            "message": "No valid RSI level"
                        },
                        "guards": {
                            "frontier": Settings.FRONTIER_INFO,
                            "stale_candle": False,
                            "spread_lock": Settings.ENFORCE_SPREAD_LOCK,
                            "dup_block": False
                        }
                    }
                    save_signal_log(partial)
                    save_trade_event({
                        "event": "SKIP_NO_VALID_LEVEL",
                        "symbol": Settings.SYMBOL,
                        "cluster_key": cluster_key
                    })
                    log_warn("No valid RSI level 1..10.")
                    time.sleep(Settings.CHECK_EVERY_SEC)
                    continue

                # 7) Duplicate guard
                blocked, block_info = _dup_block(
                    cluster_key=cluster_key,
                    state=core["state"],
                    level=int(core["level"]),
                    symbol=Settings.SYMBOL,
                    direction=core["signal"]
                )

                mt5_data = get_mt5_status(Settings.SYMBOL)

                signal_log = {
                    **core,
                    "timestamp": ts_utc(),
                    "symbol": Settings.SYMBOL,
                    "cluster_id": int(cluster_id),
                    "cluster_key": cluster_key,
                    "cluster_meta": cluster_meta,
                    "mt5": mt5_data,
                    "guards": {
                        "frontier": Settings.FRONTIER_INFO,
                        "stale_candle": False,
                        "spread_lock": Settings.ENFORCE_SPREAD_LOCK,
                        "dup_block": bool(blocked)
                    }
                }

                if blocked:
                    exec_mode, exec_info = "SKIP", block_info
                    save_trade_event({
                        "event": "SKIP_DUP_STORY_NON_IMPROVED",
                        "symbol": Settings.SYMBOL,
                        "cluster_key": cluster_key,
                        "state": core["state"],
                        "level": int(core["level"]),
                        **block_info
                    })
                    log_warn("Blocked: duplicate story & non-improved level.")
                else:
                    # 8) Execution (REAL / PAPER / SKIP)
                    exec_mode, exec_info = place_order_or_paper(signal_log)

                signal_log["order"] = {**exec_info, "mode": exec_mode}
                save_signal_log(signal_log)

                # 9) Update story meta if REAL done
                if (exec_mode == "REAL" and
                    str(signal_log["order"].get("exec_status", "")).startswith("DONE")):
                    _save_last_trade_meta({
                        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "cluster_key": cluster_key,
                        "state": signal_log["state"],
                        "level": int(signal_log["level"]),
                        "symbol": Settings.SYMBOL,
                        "direction": signal_log["signal"]
                    })

                print(
                    f"[{signal_log['timestamp']}] {Settings.SYMBOL:<7} "
                    f"| Cluster:{signal_log['cluster_id']} "
                    f"| State:{signal_log['state']:<15} "
                    f"| Lvl:{signal_log['level']} "
                    f"| Spread(pts):{mt5_data.get('spread_points', '-'):>6} "
                    f"| Signal:{signal_log['signal']:<4} "
                    f"| Mode:{exec_mode:<5} "
                    f"| ExecStatus:{signal_log['order'].get('exec_status', '-')} "
                    f"| Reason:{signal_log.get('reason', '-')}",
                    flush=True
                )

                time.sleep(Settings.CHECK_EVERY_SEC)

            except KeyboardInterrupt:
                log_info("Stopped by user.")
                break
            except Exception as e:
                log_err(f"Main loop error {type(e).__name__}: {e}")
                traceback.print_exc()
                time.sleep(10)
    finally:
        mt5.shutdown()
        log_info("MT5 shutdown.")

# ============================
# Main
# ============================

if __name__ == "__main__":
    go_live_loop()
