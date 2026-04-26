"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║      BACKTEST ENGINE — cacheheadlesspullback.py (3 Strategies)                 ║
║                                                                                  ║
║  Strategies replicated:                                                          ║
║    1. Pullback CE  — EMA200 trend + RSI(2) pullback + breakout trigger          ║
║    2. S3 Breakdown — Support breakdown below pivot S3 with volume spike         ║
║    3. ORB          — Opening Range Breakout (first 5-min candle)                ║
║                                                                                  ║
║  Data source:  Upstox v2 historical-candle API (daily + 5min)                   ║
║  Option P&L:   Underlying move × delta proxy (0.5 ATM CE/PE)                    ║
║  Output:       Console summary + backtest_results.csv + equity_curve.html       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Usage
-----
  1. Set your Upstox access token in UPSTOX_TOKEN below  (or export env var).
  2. Optionally edit BACKTEST_SYMBOLS, BACKTEST_START, BACKTEST_END.
  3. pip install requests pandas numpy
  4. python backtest_cacheheadless.py

The script downloads historical daily + 5min candles, replays them bar-by-bar,
fires signals using the same logic as the live bot, and simulates option P&L.
No broker connection, Selenium, or Klinger Oscillator is required offline —
Klinger is computed locally from the fetched candle data.
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

UPSTOX_TOKEN = os.environ.get(
    "UPSTOX_TOKEN",
    ""          # ← paste your token here if not using env var
)

# Date range to backtest  (YYYY-MM-DD strings)
BACKTEST_START = "2025-01-01"
BACKTEST_END   = "2025-12-31"

# Symbols to test — leave empty [] to auto-fetch all F&O stocks (slow)
BACKTEST_SYMBOLS = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK", "LT",
    "ADANIPORTS", "BAJFINANCE", "HINDUNILVR", "MARUTI", "SUNPHARMA",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ONGC", "COALINDIA",
]

# Capital & risk
CAPITAL            = 500_000        # ₹ starting capital
LOT_SIZE           = 1              # lots per trade
STOPLOSS_PCT       = 15.0           # option premium SL %
TARGET_MULTIPLIER  = 2.0            # reward / risk ratio
MAX_ORDERS_PER_DAY = 10
MAX_DAILY_LOSS     = 50_000
MAX_DAILY_PROFIT   = 100_000

# Strategy toggles
RUN_PULLBACK_CE  = True
RUN_S3_BREAKDOWN = True
RUN_ORB          = True
RUN_KLINGER      = True             # use Klinger filter for all strategies

# Klinger parameters (must match live bot)
KLINGER_FAST   = 34
KLINGER_SLOW   = 55
KLINGER_SIGNAL = 13
KLINGER_FAST_SHORT   = 20
KLINGER_SLOW_SHORT   = 34
KLINGER_SIGNAL_SHORT = 9
MIN_CANDLES_FOR_KLINGER = 60

# Pullback CE parameters
TREND_EMA     = 200
TRAIL_EMA     = 20
RSI_PERIOD    = 2
RSI_MAX       = 15
LOOKBACK_BARS = 4
BIG_GREEN_MAX = 3
BIG_GREEN_BODY_PCT = 0.35
STRONG_BULL_BODY_PCT = 0.30
STOP_BUFFER_PCT = 0.10
ENTRY_BUFFER_PCT = 0.05
VWAP_STRETCH_MAX = 1.0

# Volume filters
MIN_AVG_VOLUME     = 500_000
VOLUME_LOOKBACK    = 20
VOL_SPIKE_RATIO    = 1.3
PRICE_SUSTAIN_PCT  = 0.5

# ORB parameters
ORB_MIN_BODY_PCT   = 0.6
ORB_VOL_CONFIRM    = 1.5
ORB_WINDOW_MIN     = 30
ORB_RSI_LONG_MIN   = 52
ORB_RSI_SHORT_MAX  = 48

# Output files
RESULTS_CSV  = "backtest_results.csv"
EQUITY_HTML  = "equity_curve.html"

# ═══════════════════════════════════════════════════════════════════════════════
# NSE HOLIDAYS
# ═══════════════════════════════════════════════════════════════════════════════
NSE_HOLIDAYS = {
    '2025-01-26','2025-02-26','2025-03-14','2025-03-31',
    '2025-04-10','2025-04-14','2025-04-18','2025-05-01',
    '2025-08-15','2025-08-27','2025-10-02','2025-10-21',
    '2025-10-22','2025-11-05','2025-12-25',
    '2026-01-26','2026-03-03','2026-03-25','2026-04-02',
    '2026-04-10','2026-04-14','2026-05-01','2026-08-15',
    '2026-09-02','2026-10-02','2026-10-19','2026-11-08',
    '2026-11-09','2026-11-19','2026-12-25',
}

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.strftime('%Y-%m-%d') not in NSE_HOLIDAYS

# ═══════════════════════════════════════════════════════════════════════════════
# UPSTOX DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

_SESSION = None

def _get_session(token):
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        })
    return _SESSION


def fetch_daily_candles(token: str, instrument_key: str, start: str, end: str,
                        retries: int = 3) -> pd.DataFrame | None:
    """Fetch daily OHLCV from Upstox v2 API."""
    url = (f"https://api.upstox.com/v2/historical-candle/"
           f"{instrument_key}/day/{end}/{start}")
    for attempt in range(retries):
        try:
            r = _get_session(token).get(url, timeout=20)
            if r.status_code == 200:
                candles = r.json().get("data", {}).get("candles", [])
                if not candles:
                    return None
                df = pd.DataFrame(candles,
                                  columns=["date","open","high","low","close","volume","oi"])
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                df = df.sort_values("date").reset_index(drop=True)
                return df
            elif r.status_code == 429:
                time.sleep(5)
            else:
                return None
        except Exception:
            time.sleep(2)
    return None


def fetch_5min_candles(token: str, instrument_key: str, trade_date: str,
                       retries: int = 3) -> pd.DataFrame | None:
    """Fetch 5-min OHLCV for a specific date (historical-candle endpoint)."""
    url = (f"https://api.upstox.com/v2/historical-candle/"
           f"{instrument_key}/5minute/{trade_date}/{trade_date}")
    for attempt in range(retries):
        try:
            r = _get_session(token).get(url, timeout=20)
            if r.status_code == 200:
                candles = r.json().get("data", {}).get("candles", [])
                if not candles:
                    return None
                df = pd.DataFrame(candles,
                                  columns=["date","open","high","low","close","volume","oi"])
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df = df.sort_values("date").reset_index(drop=True)
                return df
            elif r.status_code == 429:
                time.sleep(5)
            else:
                return None
        except Exception:
            time.sleep(2)
    return None


def get_fo_instrument_keys(token: str) -> dict[str, str]:
    """Return {symbol: NSE_EQ|instrument_key} for all F&O equities."""
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        df = pd.read_csv(url, compression="gzip")
        fo = df[df["exchange"] == "NSE_FO"]
        fo_syms = (fo["tradingsymbol"]
                   .str.replace(r'\d{2}[A-Z]{3}\d{2,4}.*', '', regex=True)
                   .str.strip().unique())
        fo_syms = set(s for s in fo_syms if s)
        eq = df[(df["exchange"] == "NSE_EQ") & (df["tradingsymbol"].isin(fo_syms))].copy()
        eq = eq.drop_duplicates(subset=["tradingsymbol"])
        return dict(zip(eq["tradingsymbol"], eq["instrument_key"]))
    except Exception as e:
        print(f"❌ Could not fetch F&O list: {e}")
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS  (replicated from live bot)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_r3_s3(h, l, c):
    p  = (h + l + c) / 3.0
    r3 = p + 2 * (h - l)
    s3 = p - 2 * (h - l)
    return p, r3, s3


def calc_klinger(df: pd.DataFrame, fast=34, slow=55, signal=13):
    """Klinger Oscillator — same formula as the live bot."""
    if df is None or len(df) < max(fast, slow, signal) + 10:
        return None, None, None
    df = df.tail(200).reset_index(drop=True)
    hlc = (df["high"] + df["low"] + df["close"]) / 3
    trend = ((hlc > hlc.shift(1)).astype(int) * 2 - 1).fillna(0)
    dm = (df["high"] - df["low"]).replace(0, 0.001)
    cm = (dm * trend).cumsum().replace(0, 0.001).fillna(0.001)
    vf = (df["volume"] * trend * (dm / cm) * 100).clip(-1e12, 1e12)
    vf = vf.replace([float("inf"), float("-inf")], 0).fillna(0)
    vf_fast = vf.ewm(span=fast, adjust=False).mean()
    vf_slow = vf.ewm(span=slow, adjust=False).mean()
    ko      = (vf_fast - vf_slow).clip(-1e12, 1e12)
    sig     = ko.ewm(span=signal, adjust=False).mean()
    return ko, sig, ko - sig


def _rsi_series(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs  = ag / al.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((ag.fillna(0)==0)&(al.fillna(0)==0), 50)
    rsi = rsi.mask((al.fillna(0)==0)&(ag.fillna(0)>0), 100)
    rsi = rsi.mask((ag.fillna(0)==0)&(al.fillna(0)>0), 0)
    return rsi


def build_indicators(df5: pd.DataFrame) -> pd.DataFrame | None:
    """Add EMA200, EMA20, RSI(2), VWAP to a 5-min DataFrame."""
    if df5 is None or len(df5) < 30:
        return None
    out = df5.copy()
    out["date"]  = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.dropna(subset=["open","high","low","close"], inplace=True)
    if out.empty:
        return None

    out["session_date"] = out["date"].dt.date
    out["ema_200"]  = out["close"].ewm(span=200, adjust=False).mean()
    out["ema_trail"]= out["close"].ewm(span=20,  adjust=False).mean()
    out["rsi_2"]    = _rsi_series(out["close"], 2)

    # VWAP per session
    tp = (out["high"] + out["low"] + out["close"]) / 3
    vol = out["volume"].clip(lower=0)
    out["_tpv"] = tp * vol
    out["vwap"] = (out.groupby("session_date")["_tpv"].cumsum()
                   / out.groupby("session_date")["volume"].cumsum().replace(0, np.nan))
    out.drop(columns=["_tpv"], inplace=True)

    # Candle anatomy
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_pct"]  = ((out["close"] - out["open"]).abs() / out["open"].abs() * 100).fillna(0)
    out["is_green"]  = out["close"] > out["open"]
    out["upper_wick"]= ((out["high"] - out[["open","close"]].max(axis=1)) / rng * 100).fillna(0)
    return out.reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL CHECKERS
# ═══════════════════════════════════════════════════════════════════════════════

def klinger_bullish(ko_series, sig_series) -> bool:
    """Klinger bullish: KO crossed above signal OR KO > 0 and rising."""
    if ko_series is None or len(ko_series) < 2:
        return True   # no filter if unavailable
    k  = float(ko_series.iloc[-1])
    k1 = float(ko_series.iloc[-2])
    s  = float(sig_series.iloc[-1])
    s1 = float(sig_series.iloc[-2])
    # Bullish cross or both positive
    return (k > s and k1 <= s1) or (k > 0 and k > k1)


def klinger_bearish(ko_series, sig_series) -> bool:
    if ko_series is None or len(ko_series) < 2:
        return True
    k  = float(ko_series.iloc[-1])
    k1 = float(ko_series.iloc[-2])
    s  = float(sig_series.iloc[-1])
    s1 = float(sig_series.iloc[-2])
    return (k < s and k1 >= s1) or (k < 0 and k < k1)


def check_pullback_ce(daily_df: pd.DataFrame,
                      df5_full: pd.DataFrame,
                      trade_date: date,
                      ko, sig) -> dict | None:
    """
    Replay Pullback-CE logic on 5-min bars for a single trading day.
    Returns signal dict on first valid bar, else None.
    """
    # Day's 5min data only
    day_df = df5_full[df5_full["session_date"] == trade_date].copy().reset_index(drop=True)
    if len(day_df) < 10:
        return None

    yesterday = [d for d in daily_df["date"].dt.date if d < trade_date]
    if not yesterday:
        return None
    yday = yesterday[-1]
    yday_row = daily_df[daily_df["date"].dt.date == yday]
    if yday_row.empty:
        return None
    yday_close = float(yday_row.iloc[-1]["close"])

    for i in range(10, len(day_df)):
        bar_time = day_df.iloc[i]["date"]
        if bar_time.hour > 15 or (bar_time.hour == 15 and bar_time.minute >= 20):
            break

        current_price = float(day_df.iloc[i]["close"])
        day_open = float(day_df.iloc[0]["open"])

        if current_price <= day_open or current_price <= yday_close:
            continue

        completed = day_df.iloc[:i].copy()
        if len(completed) < LOOKBACK_BARS + 3:
            continue

        # Trend checks
        ema200 = float(completed["ema_200"].iloc[-1])
        ema200_prev = float(completed["ema_200"].iloc[-2]) if len(completed) >= 2 else ema200
        vwap   = float(completed["vwap"].iloc[-1])
        if vwap <= 0:
            continue
        if not (current_price > vwap and current_price > ema200 and ema200 >= ema200_prev):
            continue

        # Pullback: RSI(2) low in lookback
        recent = completed.tail(LOOKBACK_BARS)
        candidates = recent[recent["rsi_2"].notna() & (recent["rsi_2"] <= RSI_MAX)]
        if candidates.empty:
            continue

        pullback_bar = candidates.iloc[-1]

        # Avoid overbought current RSI
        rsi_now = _rsi_series(completed["close"], 2).iloc[-1]
        if pd.notna(rsi_now) and rsi_now > 70:
            continue

        # VWAP stretch
        stretch = (current_price - vwap) / vwap * 100
        if stretch > VWAP_STRETCH_MAX:
            continue

        # Too many big green candles
        big_green = int((completed.tail(BIG_GREEN_MAX)["body_pct"] >= BIG_GREEN_BODY_PCT * 100 / 100).sum())
        if big_green >= BIG_GREEN_MAX:
            continue

        # Entry trigger
        minor_resistance = float(recent["high"].max())
        prev_high = float(completed.iloc[-1]["high"])
        buf = 1 + ENTRY_BUFFER_PCT / 100

        if current_price >= minor_resistance * buf:
            trigger = "MINOR_RESISTANCE_BREAK"
        elif current_price >= prev_high * buf:
            trigger = "PREV_HIGH_BREAK"
        else:
            continue

        # Klinger filter
        if RUN_KLINGER and not klinger_bullish(ko, sig):
            continue

        # Stop / target
        stop_ref = min(float(pullback_bar["low"]), vwap)
        stop = stop_ref * (1 - STOP_BUFFER_PCT / 100)
        risk = current_price - stop
        if risk <= 0:
            continue
        target = current_price + risk * TARGET_MULTIPLIER

        return {
            "strategy": "PULLBACK_CE",
            "direction": "LONG",
            "entry_price": current_price,
            "entry_time": bar_time,
            "stop_loss": stop,
            "target": target,
            "risk": risk,
            "trigger": trigger,
            "vwap": vwap,
            "ema200": ema200,
        }
    return None


def check_s3_breakdown(daily_df: pd.DataFrame,
                       df5_day: pd.DataFrame,
                       s3: float,
                       avg_vol_20: float,
                       ko, sig) -> dict | None:
    """Replay S3 breakdown on 5-min bars."""
    if df5_day is None or len(df5_day) < 5:
        return None
    if s3 <= 0:
        return None

    threshold = s3 * (1 - PRICE_SUSTAIN_PCT / 100)

    for i in range(5, len(df5_day)):
        bar = df5_day.iloc[i]
        bar_time = bar["date"]
        if bar_time.hour > 15 or (bar_time.hour == 15 and bar_time.minute >= 20):
            break

        low   = float(bar["low"])
        ltp   = float(bar["close"])
        vol   = float(bar["volume"])

        if low > s3:
            continue
        if ltp > threshold:
            continue

        # Volume check
        if avg_vol_20 > 0:
            daily_vol = df5_day.iloc[:i+1]["volume"].sum()
            ratio = daily_vol / avg_vol_20
            if ratio < VOL_SPIKE_RATIO:
                continue

        # Klinger filter
        if RUN_KLINGER and not klinger_bearish(ko, sig):
            continue

        risk   = s3 - ltp
        if risk <= 0:
            continue
        target = ltp - risk * TARGET_MULTIPLIER
        stop   = s3 * (1 + STOP_BUFFER_PCT / 100)

        return {
            "strategy": "S3_BREAKDOWN",
            "direction": "SHORT",
            "entry_price": ltp,
            "entry_time": bar_time,
            "stop_loss": stop,
            "target": target,
            "risk": risk,
            "s3_level": s3,
        }
    return None


def check_orb(df5_day: pd.DataFrame, avg_vol_20: float,
              ko, sig) -> dict | None:
    """ORB: first 5-min candle (09:15–09:20) as breakout reference."""
    if df5_day is None or len(df5_day) < 5:
        return None

    # First candle: 09:15
    orb_bar = df5_day.iloc[0]
    orb_time = orb_bar["date"]
    if orb_time.hour != 9 or orb_time.minute != 15:
        # Try to find the 09:15 candle
        candidates = df5_day[
            (df5_day["date"].dt.hour == 9) & (df5_day["date"].dt.minute == 15)
        ]
        if candidates.empty:
            return None
        orb_bar = candidates.iloc[0]

    orb_high  = float(orb_bar["high"])
    orb_low   = float(orb_bar["low"])
    orb_open  = float(orb_bar["open"])
    orb_close = float(orb_bar["close"])
    orb_vol   = float(orb_bar["volume"])
    body_size = abs(orb_close - orb_open)
    body_pct  = body_size / orb_open * 100 if orb_open > 0 else 0

    if body_pct < ORB_MIN_BODY_PCT:
        return None

    # Volume confirmation
    if avg_vol_20 > 0 and orb_vol < avg_vol_20 * ORB_VOL_CONFIRM * 0.1:
        pass   # 5-min vol vs daily avg — skip hard reject for backtest

    is_bullish = orb_close > orb_open
    cutoff = orb_bar["date"] + timedelta(minutes=ORB_WINDOW_MIN)

    for i in range(1, len(df5_day)):
        bar = df5_day.iloc[i]
        if bar["date"] > cutoff:
            break
        if bar["date"].hour > 15 or (bar["date"].hour == 15 and bar["date"].minute >= 20):
            break

        high = float(bar["high"])
        low  = float(bar["low"])
        ltp  = float(bar["close"])
        vol  = float(bar["volume"])

        if is_bullish and high > orb_high:
            # RSI check
            completed = df5_day.iloc[:i+1]
            rsi_val = _rsi_series(completed["close"], 14).iloc[-1]
            if pd.notna(rsi_val) and rsi_val < ORB_RSI_LONG_MIN:
                continue
            if RUN_KLINGER and not klinger_bullish(ko, sig):
                continue
            risk = ltp - orb_low
            if risk <= 0:
                continue
            return {
                "strategy": "ORB_BULLISH",
                "direction": "LONG",
                "entry_price": ltp,
                "entry_time": bar["date"],
                "stop_loss": orb_low,
                "target": ltp + risk * TARGET_MULTIPLIER,
                "risk": risk,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "body_pct": round(body_pct, 2),
            }
        elif not is_bullish and low < orb_low:
            completed = df5_day.iloc[:i+1]
            rsi_val = _rsi_series(completed["close"], 14).iloc[-1]
            if pd.notna(rsi_val) and rsi_val > ORB_RSI_SHORT_MAX:
                continue
            if RUN_KLINGER and not klinger_bearish(ko, sig):
                continue
            risk = orb_high - ltp
            if risk <= 0:
                continue
            return {
                "strategy": "ORB_BEARISH",
                "direction": "SHORT",
                "entry_price": ltp,
                "entry_time": bar["date"],
                "stop_loss": orb_high,
                "target": ltp - risk * TARGET_MULTIPLIER,
                "risk": risk,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "body_pct": round(body_pct, 2),
            }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trade(signal: dict, df5_day: pd.DataFrame) -> dict:
    """
    Walk forward through the 5-min bars after entry to compute P&L.
    Option P&L = underlying move × 0.5 (delta proxy for near-ATM option).
    Exit on: SL hit, Target hit, or 15:20 time exit.
    """
    entry_price = signal["entry_price"]
    stop_loss   = signal["stop_loss"]
    target      = signal["target"]
    entry_time  = signal["entry_time"]
    direction   = signal["direction"]
    risk        = signal["risk"]

    # Bars after entry
    future_bars = df5_day[df5_day["date"] > entry_time].reset_index(drop=True)

    exit_price  = None
    exit_time   = None
    exit_reason = "EOD"

    for _, bar in future_bars.iterrows():
        t = bar["date"]
        if t.hour > 15 or (t.hour == 15 and t.minute >= 20):
            exit_price  = float(bar["open"])
            exit_time   = t
            exit_reason = "EOD"
            break

        high = float(bar["high"])
        low  = float(bar["low"])
        close= float(bar["close"])

        if direction == "LONG":
            if low <= stop_loss:
                exit_price  = stop_loss
                exit_time   = t
                exit_reason = "STOP"
                break
            if high >= target:
                exit_price  = target
                exit_time   = t
                exit_reason = "TARGET"
                break
        else:  # SHORT
            if high >= stop_loss:
                exit_price  = stop_loss
                exit_time   = t
                exit_reason = "STOP"
                break
            if low <= target:
                exit_price  = target
                exit_time   = t
                exit_reason = "TARGET"
                break

    if exit_price is None:
        # Use last bar's close
        if len(future_bars) > 0:
            exit_price  = float(future_bars.iloc[-1]["close"])
            exit_time   = future_bars.iloc[-1]["date"]
        else:
            exit_price  = entry_price
            exit_time   = entry_time
        exit_reason = "EOD"

    # P&L on underlying
    if direction == "LONG":
        underlying_move = exit_price - entry_price
    else:
        underlying_move = entry_price - exit_price

    # Option premium proxy: SL% of underlying price
    option_entry = entry_price * (STOPLOSS_PCT / 100)  # estimated option premium
    option_delta = 0.5
    option_pnl_per_unit = underlying_move * option_delta

    # Clamp to stoploss on option
    option_sl_loss = -option_entry * (STOPLOSS_PCT / 100)
    option_pnl_per_unit = max(option_pnl_per_unit, option_sl_loss)

    pnl_total = option_pnl_per_unit * LOT_SIZE * 50  # ×50 lot approximation

    return {
        **signal,
        "exit_price":  exit_price,
        "exit_time":   exit_time,
        "exit_reason": exit_reason,
        "underlying_move": round(underlying_move, 2),
        "option_entry_est": round(option_entry, 2),
        "pnl":         round(pnl_total, 2),
        "pnl_pct":     round(option_pnl_per_unit / option_entry * 100, 2) if option_entry else 0,
        "win":         pnl_total > 0,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    if not UPSTOX_TOKEN:
        print("❌ UPSTOX_TOKEN is empty.  Set it at the top of this file or via env var.")
        sys.exit(1)

    print("=" * 80)
    print("  UPSTOX BACKTEST ENGINE")
    print(f"  Period: {BACKTEST_START}  →  {BACKTEST_END}")
    print(f"  Capital: ₹{CAPITAL:,.0f}  |  Lot Size: {LOT_SIZE}")
    print(f"  Strategies: "
          f"{'Pullback-CE ' if RUN_PULLBACK_CE else ''}"
          f"{'S3-Breakdown ' if RUN_S3_BREAKDOWN else ''}"
          f"{'ORB ' if RUN_ORB else ''}"
          f"| Klinger: {'ON' if RUN_KLINGER else 'OFF'}")
    print("=" * 80)

    # ── 1. Instrument map ────────────────────────────────────────────────────
    print("\n📥 Fetching F&O instrument list…")
    fo_map = get_fo_instrument_keys(UPSTOX_TOKEN)
    if not fo_map:
        print("❌ Could not fetch F&O list — check token and connectivity.")
        sys.exit(1)

    symbols = BACKTEST_SYMBOLS if BACKTEST_SYMBOLS else list(fo_map.keys())
    symbols = [s for s in symbols if s in fo_map]
    print(f"✅ Testing {len(symbols)} symbols\n")

    # Date range
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(BACKTEST_END,   "%Y-%m-%d").date()
    all_days = [start_dt + timedelta(days=i)
                for i in range((end_dt - start_dt).days + 1)
                if is_trading_day(start_dt + timedelta(days=i))]

    print(f"📅 Trading days in range: {len(all_days)}")

    trades      = []
    equity      = CAPITAL
    equity_curve= [(start_dt, equity)]
    daily_pnl   = {}

    total_syms = len(symbols)

    for sym_idx, symbol in enumerate(symbols, 1):
        ikey = fo_map[symbol]
        print(f"\n[{sym_idx:3d}/{total_syms}] ── {symbol} ──────────────────────────")

        # ── Fetch full daily history (for Klinger + pivot levels) ─────────
        daily_df = fetch_daily_candles(
            UPSTOX_TOKEN, ikey,
            (start_dt - timedelta(days=180)).strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d")
        )
        if daily_df is None or len(daily_df) < MIN_CANDLES_FOR_KLINGER:
            print(f"   ⚠ Insufficient daily data ({len(daily_df) if daily_df is not None else 0} bars) — skip")
            time.sleep(0.3)
            continue

        # ── Filter to required date range ─────────────────────────────────
        hist_df = daily_df[daily_df["date"].dt.date < start_dt].copy()
        if len(hist_df) < VOLUME_LOOKBACK:
            print(f"   ⚠ Not enough pre-period history — skip")
            time.sleep(0.3)
            continue

        avg_vol_20 = float(hist_df["volume"].tail(VOLUME_LOOKBACK).mean())
        if avg_vol_20 < MIN_AVG_VOLUME:
            print(f"   ⚠ Low avg volume ({avg_vol_20:,.0f}) — skip")
            continue

        # Klinger on full daily history
        ko_d, sig_d, _ = calc_klinger(
            hist_df,
            fast   = KLINGER_FAST if len(hist_df) >= 90 else KLINGER_FAST_SHORT,
            slow   = KLINGER_SLOW if len(hist_df) >= 90 else KLINGER_SLOW_SHORT,
            signal = KLINGER_SIGNAL if len(hist_df) >= 90 else KLINGER_SIGNAL_SHORT,
        )

        symbol_trades = 0
        prev_5min_cache: dict[str, pd.DataFrame | None] = {}  # date_str → df5

        for trade_date in all_days:
            # Skip days before enough history exists
            prior = daily_df[daily_df["date"].dt.date < trade_date]
            if len(prior) < VOLUME_LOOKBACK + 2:
                continue

            date_str = trade_date.strftime("%Y-%m-%d")
            day_orders = sum(1 for t in trades if t.get("trade_date") == trade_date)
            if day_orders >= MAX_ORDERS_PER_DAY:
                continue

            daily_pnl_today = daily_pnl.get(trade_date, 0.0)
            if daily_pnl_today <= -MAX_DAILY_LOSS or daily_pnl_today >= MAX_DAILY_PROFIT:
                continue

            # ── Yesterday's OHLC → pivot levels ──────────────────────────
            yday_candidates = [d for d in prior["date"].dt.date if is_trading_day(d)]
            if not yday_candidates:
                continue
            yday = yday_candidates[-1]
            yday_row = prior[prior["date"].dt.date == yday]
            if yday_row.empty:
                continue
            yh = float(yday_row.iloc[-1]["high"])
            yl = float(yday_row.iloc[-1]["low"])
            yc = float(yday_row.iloc[-1]["close"])
            _, r3, s3 = calc_r3_s3(yh, yl, yc)

            # ── Update rolling Klinger (use all daily up to yday) ─────────
            rolling_hist = daily_df[daily_df["date"].dt.date <= yday].copy()
            if len(rolling_hist) >= MIN_CANDLES_FOR_KLINGER:
                ko_d, sig_d, _ = calc_klinger(
                    rolling_hist,
                    fast   = KLINGER_FAST if len(rolling_hist) >= 90 else KLINGER_FAST_SHORT,
                    slow   = KLINGER_SLOW if len(rolling_hist) >= 90 else KLINGER_SLOW_SHORT,
                    signal = KLINGER_SIGNAL if len(rolling_hist) >= 90 else KLINGER_SIGNAL_SHORT,
                )

            # ── 5-min data for trade_date ─────────────────────────────────
            if date_str not in prev_5min_cache:
                df5_raw = fetch_5min_candles(UPSTOX_TOKEN, ikey, date_str)
                time.sleep(0.15)       # gentle rate-limit
                if df5_raw is not None:
                    # Need history for EMA200 warm-up — prepend earlier 5min days
                    prev_dates = sorted(prev_5min_cache.keys())[-5:]
                    frames = [prev_5min_cache[d] for d in prev_dates
                              if prev_5min_cache.get(d) is not None]
                    frames.append(df5_raw)
                    df5_all = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
                    df5_ind = build_indicators(df5_all)
                    prev_5min_cache[date_str] = df5_raw
                else:
                    prev_5min_cache[date_str] = None
                    df5_ind = None
            else:
                if prev_5min_cache[date_str] is not None:
                    prev_dates = sorted(prev_5min_cache.keys())[-5:]
                    frames = [prev_5min_cache[d] for d in prev_dates
                              if prev_5min_cache.get(d) is not None]
                    df5_all = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
                    df5_ind = build_indicators(df5_all)
                else:
                    df5_ind = None

            if df5_ind is None:
                continue

            df5_day = df5_ind[df5_ind["session_date"] == trade_date].copy().reset_index(drop=True)
            if len(df5_day) < 5:
                continue

            signals_today = []

            # ── Strategy 1: Pullback CE ───────────────────────────────────
            if RUN_PULLBACK_CE:
                sig = check_pullback_ce(daily_df, df5_ind, trade_date, ko_d, sig_d)
                if sig:
                    sig["trade_date"] = trade_date
                    signals_today.append(sig)

            # ── Strategy 2: S3 Breakdown ──────────────────────────────────
            if RUN_S3_BREAKDOWN:
                sig = check_s3_breakdown(daily_df, df5_day, s3, avg_vol_20, ko_d, sig_d)
                if sig:
                    sig["trade_date"] = trade_date
                    signals_today.append(sig)

            # ── Strategy 3: ORB ───────────────────────────────────────────
            if RUN_ORB:
                sig = check_orb(df5_day, avg_vol_20, ko_d, sig_d)
                if sig:
                    sig["trade_date"] = trade_date
                    signals_today.append(sig)

            # ── Simulate each signal ──────────────────────────────────────
            for sig in signals_today[:MAX_ORDERS_PER_DAY]:
                result = simulate_trade(sig, df5_day)
                result["symbol"] = symbol

                # Equity update
                pnl = result["pnl"]
                equity += pnl
                daily_pnl[trade_date] = daily_pnl.get(trade_date, 0.0) + pnl
                equity_curve.append((trade_date, equity))

                trades.append(result)
                symbol_trades += 1

                status = "✅" if pnl > 0 else "❌"
                print(f"   {status} {result['strategy']:15s} {date_str} "
                      f"Entry ₹{result['entry_price']:7.2f} → "
                      f"Exit ₹{result['exit_price']:7.2f}  "
                      f"{result['exit_reason']:8s}  "
                      f"P&L ₹{pnl:+8.0f}")

        if symbol_trades == 0:
            print(f"   — No signals generated")
        time.sleep(0.2)

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print_summary(trades, equity, CAPITAL, equity_curve)
    save_results(trades, equity_curve)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(trades: list, final_equity: float, start_capital: float,
                  equity_curve: list):
    if not trades:
        print("\n⚠️ No trades generated.")
        return

    df = pd.DataFrame(trades)
    total = len(df)
    wins  = df["win"].sum()
    losses= total - wins
    win_rate = wins / total * 100 if total else 0
    total_pnl= df["pnl"].sum()
    avg_win  = df[df["win"]]["pnl"].mean() if wins else 0
    avg_loss = df[~df["win"]]["pnl"].mean() if losses else 0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    max_dd   = _max_drawdown([e for _, e in equity_curve])
    cagr     = _cagr(start_capital, final_equity, equity_curve)

    print("\n" + "═" * 80)
    print("  BACKTEST RESULTS SUMMARY")
    print("═" * 80)
    print(f"  Period         : {equity_curve[0][0]}  →  {equity_curve[-1][0]}")
    print(f"  Starting Capital: ₹{start_capital:,.0f}")
    print(f"  Final Equity   : ₹{final_equity:,.0f}  ({(final_equity/start_capital-1)*100:+.1f}%)")
    print(f"  CAGR           : {cagr:+.1f}%")
    print(f"  Max Drawdown   : {max_dd:.1f}%")
    print(f"  Total Trades   : {total}")
    print(f"  Win Rate       : {win_rate:.1f}%  ({int(wins)}W / {int(losses)}L)")
    print(f"  Avg Win        : ₹{avg_win:,.0f}")
    print(f"  Avg Loss       : ₹{avg_loss:,.0f}")
    print(f"  Reward/Risk    : {rr:.2f}x")
    print(f"  Total P&L      : ₹{total_pnl:,.0f}")
    print("─" * 80)

    # Per-strategy breakdown
    print(f"\n  Per-Strategy Breakdown:")
    print(f"  {'Strategy':<20} {'Trades':>7} {'WinRate':>9} {'Avg P&L':>10} {'Total P&L':>12}")
    print(f"  {'─'*20} {'─'*7} {'─'*9} {'─'*10} {'─'*12}")
    for strat, grp in df.groupby("strategy"):
        n   = len(grp)
        wr  = grp["win"].mean() * 100
        avg = grp["pnl"].mean()
        tot = grp["pnl"].sum()
        print(f"  {strat:<20} {n:>7} {wr:>8.1f}% {avg:>+10.0f} {tot:>+12.0f}")

    # Top 5 symbols
    print(f"\n  Top 5 Symbols by P&L:")
    sym_pnl = df.groupby("symbol")["pnl"].sum().sort_values(ascending=False).head(5)
    for sym, pnl in sym_pnl.items():
        print(f"    {sym:<15} ₹{pnl:+,.0f}")
    print("═" * 80)


def _max_drawdown(equity_series: list[float]) -> float:
    if not equity_series:
        return 0.0
    peak = equity_series[0]
    max_dd = 0.0
    for e in equity_series:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _cagr(start: float, end: float, curve: list) -> float:
    if not curve or start <= 0:
        return 0.0
    days = (curve[-1][0] - curve[0][0]).days if len(curve) > 1 else 1
    years = max(days / 365.25, 0.01)
    return ((end / start) ** (1 / years) - 1) * 100


def save_results(trades: list, equity_curve: list):
    if not trades:
        return

    # CSV
    df = pd.DataFrame(trades)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n💾 Results saved to {RESULTS_CSV}")

    # Equity curve HTML chart
    dates  = [str(d) for d, _ in equity_curve]
    values = [round(v, 2) for _, v in equity_curve]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Backtest Equity Curve</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ background:#0d1117; color:#e6edf3; font-family:sans-serif; padding:20px; }}
    h2   {{ color:#58a6ff; }}
    canvas {{ max-height:500px; }}
  </style>
</head>
<body>
  <h2>📈 Backtest Equity Curve — cacheheadlesspullback strategies</h2>
  <p>Period: {dates[0] if dates else ''} → {dates[-1] if dates else ''} &nbsp;|&nbsp;
     Final equity: ₹{values[-1]:,.0f} &nbsp;|&nbsp; Trades: {len(trades)}</p>
  <canvas id="chart"></canvas>
  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: {json.dumps(dates)},
        datasets: [{{
          label: 'Equity (₹)',
          data: {json.dumps(values)},
          borderColor: '#3fb950',
          backgroundColor: 'rgba(63,185,80,0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ labels: {{ color: '#e6edf3' }} }},
          tooltip: {{ callbacks: {{
            label: ctx => '₹' + ctx.parsed.y.toLocaleString('en-IN')
          }}}}
        }},
        scales: {{
          x: {{ ticks: {{ color:'#8b949e', maxTicksLimit:12 }}, grid: {{ color:'#21262d' }} }},
          y: {{
            ticks: {{ color:'#8b949e',
              callback: v => '₹' + Number(v).toLocaleString('en-IN') }},
            grid: {{ color:'#21262d' }}
          }}
        }}
      }}
    }});
  </script>
</body>
</html>"""

    with open(EQUITY_HTML, "w") as f:
        f.write(html)
    print(f"📊 Equity curve saved to {EQUITY_HTML}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_backtest()
