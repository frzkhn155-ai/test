import os
import sys
import pickle
import time
import imaplib
import email
import re
import json
import pyperclip
import requests
import pandas as pd
import csv
import numpy as np
import threading
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── AI Assistant (Groq free tier — see ai_assistant.py for setup) ─────────────
try:
    from ai_assistant import start_ai_assistant, ai_status, AI_ENABLED as _AI_ENABLED
    _AI_IMPORT_OK = True
except ImportError:
    _AI_IMPORT_OK = False
    def start_ai_assistant(*a, **kw): pass
    def ai_status(): return "AI Assistant: ai_assistant.py not found"
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# ============ FORCE UNBUFFERED OUTPUT (fixes Pydroid3 display freeze) ============
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line-buffered
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# ============ CREDENTIALS - ENV VARS (with hardcoded fallbacks) ============
EMAIL          = os.environ.get("UPSTOX_EMAIL",    "frzkhn155@gmail.com")
EMAIL_PASSWORD = os.environ.get("UPSTOX_PASSWORD", "vdeahogzvpsmfirv")
MOBILE_NUMBER  = os.environ.get("UPSTOX_MOBILE",   "7397408750")
PASSCODE       = os.environ.get("UPSTOX_PASSCODE", "952495")

# ── Upstox OAuth app credentials (for Android token refresh) ─────────────────
# Get these from https://account.upstox.com/developer/apps → your app
# API Key    = "Client ID" on the Upstox developer portal
# API Secret = "Client Secret"
# Redirect   = must match exactly what you set in the app (use the one below)
UPSTOX_API_KEY      = os.environ.get("UPSTOX_API_KEY",    "ea9b2ade-6720-4a0b-a8a5-6e1710f55844")
UPSTOX_API_SECRET   = os.environ.get("UPSTOX_API_SECRET", "csxmppf5zd")
UPSTOX_REDIRECT_URI = "http://127.0.0.1:8080/"        # must match your app settings

# ── Headless OAuth local server port ─────────────────────────────────────────
# A tiny HTTP server listens on this port to capture the OAuth redirect code.
# Must match UPSTOX_REDIRECT_URI above (port 8080).
HEADLESS_SERVER_PORT = 8080
UPSTOX_REFRESH_TOKEN_FILE = "upstox_refresh_token.txt"
# ─────────────────────────────────────────────────────────────────────────────

# ========== CHARTINK CONFIGURATION (for 5min data) ==========
CHARTINK_BASE_URL = "https://chartink.com/oapi"
CHARTINK_COOKIES = {
    "_ga": "GA1.2.1533223166.1742236648",
    "XSRF-TOKEN": "eyJpdiI6IjFkM21JUDJhSjI3eWxVRno5TnRIcVE9PSIsInZhbHVlIjoiRzRkRlh1THBGVTFoZE5Rbm5oSjhSdG84VUo1NFJLNUs1WmtIbXNOL2IxQkZWM016TkZFVE9KRk9Ed0Z3U1VTVCsvNUw1NzM2OHZxL2JoTEE3Mkx2U2x0Q0NzdEg1eThPakYwd2tvMEhsbGZlRENGbmFHalFGbGhyV2VHL2tMTEciLCJtYWMiOiJjZmY1YTc4NmQ5MTZhZTZkZjExN2YyMTc3M2QxNzIxODYyMzhkYzIwMmJkNWM3NmRkNTRmNWMwOWNmZTNmZTc1IiwidGFnIjoiIn0=",
    "ci_session": "eyJpdiI6Ik5yWkd3UTM1N0FYbjFJcmI4NTdxWlE9PSIsInZhbHVlIjoiVDZBazRrTFdIMlRFMW52d0JFU1pSempOTndnT09jNC9tRXoxZXZwamE2RUVIQWlzQTl3b3pEa2NTYXpzQk5ZWWxEcGViUmM2ZmRBQnQxMVFFZy9SOFBBaDNScmFER3BVUWE1V21URVN3bk5IMzBNWVIyaHhmWUVsT1VDelZQVjgiLCJtYWMiOiJmNWVkN2RlMzIzNDkzNDcyZDU5Y2RhODQ5YjZjYzI4M2I0YTA0YjBhYTA4YTFkNTgwYzFjZTc5YjlmZWJiMDZiIiwidGFnIjoiIn0="
}
# ⚠️  UPDATE CHARTINK_COOKIES FROM YOUR BROWSER if 5min data fails:
# 1. Go to chartink.com/stocks-new in Chrome
# 2. DevTools (F12) → Network → refresh → click any chartink.com request
# 3. Copy XSRF-TOKEN and ci_session from Request Headers → Cookies

# ========== HARDCODED TOKEN OPTION ==========
HARDCODED_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIyMkM4REwiLCJqdGkiOiI2OWUxYTUxNmVhOTJhNTBmZTYwZjAwY2IiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzc2Mzk1NTQyLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzY0NjMyMDB9.enP-ZbfL-q4JCJ78u-l9QVDUj71LOV2Ogi8QmeLXRDg"          # Intentionally blank — token is injected via UPSTOX_TOKEN secret
USE_HARDCODED_TOKEN = True   # Disabled — bot reads upstox_token.txt written by the workflow

# Token timestamp file
TOKEN_TIMESTAMP_FILE = "token_timestamp.json"
UPSTOX_TOKEN_FILE = "upstox_token.txt"

# ========== CONFIGURATION ==========
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
MARKET_STABILIZATION_MINUTES = 5
EXIT_START_TIME = "15:20"

# Volume / Filter Settings
MIN_AVG_VOLUME = 500_000
VOLUME_SPIKE_THRESHOLD = 1.3
VOLUME_LOOKBACK_DAYS = 20
USE_DYNAMIC_VOLUME_THRESHOLD = True
MAX_WORKERS = 3
DEBUG_MODE = False                     # <-- changed to True
BATCH_SIZE = 100
MAX_INSTRUMENTS_PER_BATCH = 500

# Logging
ALERT_LOG_FILE = "r3_live_alerts.txt"
ALERT_CSV_FILE = "r3_live_alerts.csv"
EXIT_LOG_FILE = "exits_log.txt"
EXIT_CSV_FILE = "exits_log.csv"
POSITION_LOG_FILE = "positions_tracking.csv"

# AUTOMATED TRADING CONFIGURATION
ENABLE_AUTO_TRADING = True
ORDER_QUANTITY = 1
ORDER_PRODUCT = 'D'                   # <-- changed from 'I' to 'D' (NRML for options)
PLACE_STOPLOSS = True
STOPLOSS_PERCENTAGE = 15.0
MAX_ORDERS_PER_DAY = 10
MIN_ORDER_GAP_SECONDS = 300
ORDER_VERIFICATION_DELAY = 3

# TEST MODE (FALSE = normal market hours)
TEST_MODE = False
BYPASS_MARKET_CHECKS = TEST_MODE      # used in is_market_open / is_market_stabilized

# When TEST_MODE=True and the bot starts before 05:30 IST, wait until the
# Upstox order window opens rather than running and generating rejected orders.
# Set False to scan immediately without waiting (signals logged, orders blocked).
WAIT_FOR_ORDER_WINDOW = True

# EXIT STRATEGY CONFIGURATION
ENABLE_EXIT_MANAGEMENT = True
MAX_DAILY_LOSS = 50000
MAX_DAILY_PROFIT = 100000
ENABLE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION = 50.0
TRAILING_STOP_PERCENTAGE = 10.0
TARGET_PROFIT_MULTIPLIER = 2.0
ENABLE_TIME_BASED_EXIT = True
ENABLE_EXPIRY_DAY_EXIT = True
EXPIRY_EXIT_TIME = "15:00"
ENABLE_STRATEGY_EXITS = True
POSITION_MONITORING_INTERVAL = 30

# TREND PULLBACK CE STRATEGY
ENABLE_PULLBACK_CE_STRATEGY = True
TREND_FILTER_EMA_PERIOD = 200
PULLBACK_TRAIL_EMA_PERIOD = 20
PULLBACK_RSI_PERIOD = 2
PULLBACK_RSI_MAX = 15
PULLBACK_LOOKBACK_BARS = 4
PULLBACK_ENTRY_BUFFER_PERCENT = 0.05
PULLBACK_VWAP_STRETCH_PERCENT = 1.0
PULLBACK_MAX_BIG_GREEN_CANDLES = 3
PULLBACK_BIG_GREEN_BODY_PERCENT = 0.35
PULLBACK_STRONG_BULL_BODY_PERCENT = 0.30
PULLBACK_STOP_BUFFER_PERCENT = 0.10
PULLBACK_OPTION_SAFETY_SL_PERCENT = 25.0
PULLBACK_INTRADAY_VOLUME_LOOKBACK = 10
PULLBACK_INTRADAY_VOLUME_RATIO_MIN = 0.80

# KLINGER OSCILLATOR CONFIGURATION
ENABLE_KLINGER_FILTER = True
KLINGER_FAST = 34
KLINGER_SLOW = 55
KLINGER_SIGNAL = 13
KLINGER_PAPER_MODE = False
ENABLE_KLINGER_FOR_BOX = True
ENABLE_KLINGER_FOR_RANGE = True

# ============ CANDLE CACHE CONFIGURATION ============
ENABLE_CANDLE_CACHE = True
CACHE_DIRECTORY = "candle_cache"
CACHE_EXPIRY_DAYS = 7  # Re-fetch if cache older than this
MIN_CANDLES_FOR_KLINGER = 60  # Minimum candles required (reduced from 90)
ADAPTIVE_KLINGER_LOOKBACK = True  # Use shorter periods for limited data
KLINGER_FAST_SHORT = 20  # For 60-89 days of data
KLINGER_SLOW_SHORT = 34
KLINGER_SIGNAL_SHORT = 9
CACHE_UPDATE_HOUR = 18  # Update cache after market close (6 PM)
CACHE_STATS_FILE = "cache_stats.json"

# ========== FII/DII + ORB CONFIGURATION ==========
ENABLE_FII_DII_FILTER = True
FII_DII_URL = "https://munafasutra.com/nse/FIIDII/"
FII_DII_UPDATE_INTERVAL = 86400  # Fetch once per day — FII/DII data is published once after market close
FII_DII_CACHE_FILE = "fii_dii_cache.json"

# ── FII/DII MULTI-DAY TREND ANALYSIS ─────────────────────────────────────────
# Reads historical FII_DII_YYYYMMDD.csv files to detect institutional patterns:
#   STRONG_ACCUMULATION : Both FII cash + FNO bought (most bullish)
#   FII_BUY_DII_SELL    : FII cash bought, FNO sold (FII leading — bullish lean)
#   FII_SELL_DII_BUY    : FII cash sold, FNO bought (DII support — caution)
#   UNUSUAL_CHANGE      : Reversed from previous day (both sold -> both bought etc.)
ENABLE_FII_DII_TREND_FILTER  = True   # Master switch for trend-based adjustments
FII_DII_TREND_CACHE_FILE     = "fii_dii_trend_cache.json"
# Volume threshold relief for strong-accumulation stocks (10% easier to pass)
FII_DII_TREND_VOLUME_RELIEF  = 0.90   # multiply thr by this for strong accumulation
# Confidence score adjustments (added to base score for sorting/logging)
FII_DII_SCORE_STRONG_ACC     = +2     # both bought today
FII_DII_SCORE_FII_BUY        = +1     # FII cash bought, FNO sold
FII_DII_SCORE_FII_SELL       = -1     # FII cash sold, FNO bought
FII_DII_SCORE_UNUSUAL        = +2     # sudden reversal — high conviction move

# ORB STRATEGY CONFIGURATION
ENABLE_ORB_STRATEGY = True
ORB_TIMEFRAME_MINUTES = 5  # First 5 minutes (9:15-9:20)
ORB_MIN_CANDLE_BODY_PERCENT = 0.5  # Minimum 0.5% body size
ORB_VOLUME_CONFIRMATION = 1.5  # Raised: 1.5x average volume required (was 1.2x)
ORB_BREAKOUT_WINDOW_MINUTES = 30  # Tightened: only trade within 30 min of 9:20 (was 60)
ORB_TARGET_MULTIPLIER = 2.0  # Target = 2x candle body
ORB_STOP_MULTIPLIER = 1.0  # Stop at opposite end of candle
ORB_MIN_VOLUME = 500000  # Minimum average volume
ORB_ENABLE_MARKET_ALIGNMENT = True  # Check Nifty direction
ORB_ENABLE_FII_DII_FILTER = True  # Only trade with FII/DII alignment

# ORB QUALITY GATE — Klinger + RSI secondary filter
ORB_ENABLE_KLINGER_GATE   = True   # Require Klinger alignment for ORB signals
ORB_ENABLE_RSI_GATE       = True   # Require RSI momentum confirmation
ORB_RSI_LONG_MIN          = 52     # LONG ORB: RSI must be >= this (momentum present)
ORB_RSI_SHORT_MAX         = 48     # SHORT ORB: RSI must be <= this (momentum present)
ORB_MIN_CANDLE_BODY_LONG  = 0.6    # LONG ORB: slightly higher body % required
ORB_MIN_CANDLE_BODY_SHORT = 0.6    # SHORT ORB: slightly higher body % required
ORB_REQUIRE_STRONG_FII_FOR_MEDIUM_RSI = True  # If RSI borderline, require STRONG FII/DII

# File paths for ORB logging
ORB_SIGNALS_FILE = "orb_signals.csv"
ORB_TRADES_FILE = "orb_trades.csv"
ORB_LOG_FILE = "orb_trading_log.txt"

# ============ OPTION TRADING CONFIGURATION ============
OPTION_PREMIUM_MIN_THRESHOLD = 1.0  # Minimum premium to consider
OPTION_PREMIUM_MAX_THRESHOLD = 500.0  # Maximum premium to consider
OPTION_LTP_RETRY_ATTEMPTS = 5  # Number of retries for LTP fetch
OPTION_FALLBACK_PREMIUM_ENABLED = True  # Use estimated premium if LTP fails

# Entry Types
ENTRY_BREAKOUT = "BREAKOUT"
ENTRY_PULLBACK = "PULLBACK"
ENTRY_SQUEEZE = "SQUEEZE"
ENTRY_ORB_BULLISH = "ORB_BULLISH"
ENTRY_ORB_BEARISH = "ORB_BEARISH"

# Exit Types
EXIT_TARGET = "TARGET"
EXIT_STOP = "STOP"
EXIT_TRAILING = "TRAILING"
EXIT_REVERSAL = "REVERSAL"

# ── ALERTED STOCK SETS ───────────────────────────────────────────────────────
# R3/S3 breakout
R3_ALERTED_STOCKS         = set()   # fired R3 LONG today
S3_ALERTED_STOCKS         = set()   # fired S3 SHORT today
ALERTED_STOCKS            = set()   # legacy alias (R3 ∪ S3)
# ─────────────────────────────────────────────────────────────────────────────
R3_LEVELS = {}
SYMBOL_TO_ISIN = {}
ISIN_TO_SYMBOL = {}
SYMBOL_TO_FO_KEY = {}  # symbol -> NSE_FO instrument_key (fallback for Upstox 5min endpoints)
VOLUME_DATA = {}
INITIALIZATION_RETRIES = 0
OPTIONS_CACHE = {}
DAILY_ORDER_COUNT = 0
LAST_ORDER_TIME = {}
PLACED_ORDERS = {}

# EXIT MANAGEMENT GLOBALS
ACTIVE_POSITIONS = {}
DAILY_PNL = 0.0
CLOSED_POSITIONS = []
TRADING_STOPPED = False
POSITION_PEAK_PRICES = {}
POSITION_TRAILING_SL  = {}   # Tracks the last SL trigger_price pushed to the broker per position

# ============ REAL-TIME 5MIN CANDLE BUILDER GLOBALS ============
from collections import defaultdict
import logging as _logging

# ── STRUCTURED LOGGER (replaces raw print for debug-level noise) ─────────────
# Set DEBUG_MODE=True  → logger emits DEBUG+INFO; set False → INFO only.
# All existing `if DEBUG_MODE: print(...)` blocks are preserved unchanged —
# they naturally suppress when DEBUG_MODE=False.  This logger is for new code.
_logger = _logging.getLogger("upstox_bot")
_handler = _logging.StreamHandler()
_handler.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
_logger.addHandler(_handler)
_logger.setLevel(_logging.DEBUG)   # controlled at call site via DEBUG_MODE check
# ─────────────────────────────────────────────────────────────────────────────
REALTIME_CANDLES = defaultdict(list)      # symbol -> list of completed 5min candles
CURRENT_CANDLE = {}                        # symbol -> {open, high, low, close, volume, candle_start}
CANDLE_BUILDER_LOCK = threading.Lock()

# MARGIN CHECK CACHE (avoids excessive API calls; refreshed at most once per minute)
_CACHED_AVAILABLE_MARGIN = None
_MARGIN_CACHE_TIME = None
_MARGIN_CACHE_LOCK = threading.Lock()
_MARGIN_CACHE_TTL_SECONDS = 60  # seconds

# FII/DII GLOBALS
FII_DII_DATA = {}
FII_DII_LAST_UPDATE = None
FII_DII_STRONG_BUY = set()
FII_DII_STRONG_SELL = set()
FII_DII_MIXED = set()

# ── FII/DII MULTI-DAY TREND SETS ─────────────────────────────────────────────
FII_DII_TREND_STRONG_ACCUMULATION = set()   # Both bought today
FII_DII_TREND_FII_BUY_DII_SELL    = set()   # FII cash bought, FNO sold
FII_DII_TREND_FII_SELL_DII_BUY    = set()   # FII cash sold, FNO bought
FII_DII_TREND_UNUSUAL_CHANGE      = set()   # Reversed vs previous day
FII_DII_TREND_LOCK                = threading.RLock()  # Thread safety

# ORB GLOBALS
ORB_CANDLES = {}
ORB_SIGNALS = {}
ORB_LATE_CHECKED = set()   # symbols confirmed zero-volume at 09:20; retry until volume appears
ORB_ACTIVE_TRADES = {}
ORB_ALERTED_STOCKS = set()   # fired ORB signal today
ORB_ORDER_COUNT = 0
ORB_PROCESSED_TODAY = False

# ── PERSISTENT HTTP SESSIONS ─────────────────────────────────────────────────
# Reusing a Session keeps the TCP/TLS connection alive across calls.
# DNS + TLS handshake costs 200-400 ms per new connection; with a persistent
# session that drops to <5 ms on subsequent calls (40-60% total latency saving).
_UPSTOX_SESSION        = None   # requests.Session for Upstox API
_UPSTOX_SESSION_TOKEN  = ""     # tracks which Bearer token the session was built for
# _CHARTINK_SESSION is created near the ChartInk fetch helpers (further below)
# ─────────────────────────────────────────────────────────────────────────────

# CACHE GLOBALS
CANDLE_CACHE = {}
CACHE_STATS = {
    'cache_hits': 0,
    'cache_misses': 0,
    'api_calls_saved': 0,
    'total_cached_symbols': 0,
    'last_updated': None
}

# ── INTRADAY CANDLE CACHE (5min) ──────────────────────────────────────────────
_5MIN_CACHE:         dict = {}
_5MIN_CACHE_TTL_S    = 28
# ── INTRADAY CANDLE CACHE (15min) ─────────────────────────────────────────────
_15MIN_CACHE:        dict = {}
_15MIN_CACHE_TTL_S   = 55
_INTRADAY_CACHE_LOCK = threading.Lock()

# ORDER REJECTION TRACKING — tracks signals that fired but couldn't place (service hours, limits etc.)
REJECTED_ORDER_SIGNALS = []   # list of {'symbol', 'strategy', 'reason', 'timestamp'}

# ENHANCED FALSE ALERT PREVENTION VARIABLES
LAST_BREAKOUT_STATE = {}

BREACH_CONFIRMATION_CYCLES = 2
BREACH_TIME_WINDOW = 180    # Extended: 90s too short (only 3 scans), now 6 scans to confirm
PRICE_SUSTAINABILITY_PERCENT = 0.5  # 0.2% was too tight (₹0.66 on ₹330 stock = noise level)

# Option cache
OPTION_CHAIN_CACHE = {}
OPTION_CHAIN_CACHE_EXPIRY = 300  # 5 minutes in seconds

# NSE Holidays 2025
NSE_HOLIDAYS_2025 = {
    '2025-01-26', '2025-02-26', '2025-03-14', '2025-03-31',
    '2025-04-10', '2025-04-14', '2025-04-18', '2025-05-01',
    '2025-08-15', '2025-08-27', '2025-10-02', '2025-10-21',
    '2025-10-22', '2025-11-05', '2025-12-25',
}

# NSE Holidays 2026
NSE_HOLIDAYS_2026 = {
    '2026-01-26',  # Republic Day
    '2026-03-03',  # Holi
    '2026-03-25',  # Gudi Padwa
    '2026-04-02',  # Mahavir Jayanti
    '2026-04-10',  # Good Friday
    '2026-04-14',  # Dr. Ambedkar Jayanti
    '2026-05-01',  # Maharashtra Day
    '2026-08-15',  # Independence Day
    '2026-09-02',  # Ganesh Chaturthi
    '2026-10-02',  # Gandhi Jayanti
    '2026-10-19',  # Dussehra
    '2026-11-08',  # Diwali
    '2026-11-09',  # Diwali Balipratipada
    '2026-11-19',  # Gurunanak Jayanti
    '2026-12-25',  # Christmas
}

# Combine both years
NSE_HOLIDAYS = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026

pd.options.mode.chained_assignment = None

# ============================================================================
# CANDLE CACHE MANAGEMENT SYSTEM
# ============================================================================

# ============ 5MIN DATA FAILURE TRACKING ============
# Instruments that persistently return 400 for 5min data are blacklisted after MAX_5MIN_FAILURES
FAST_TRADE_5MIN_FAILURES = {}          # instrument_key -> failure_count
MAX_5MIN_FAILURES = 3                   # blacklist after this many consecutive failures
FAST_TRADE_5MIN_BLACKLIST = set()       # instrument_keys permanently skipped for 5min data

# ── ChartInk historical base cache ───────────────────────────────────────────
# Fetched ONCE per symbol (on first call), cached in memory.
# Real-time LTP ticks from update_realtime_candle() are merged on top,
# giving a complete dataset from the very first scan at 09:15.
_CK_HIST_CACHE: dict = {}               # symbol -> DataFrame (historical OHLCV base)
_CK_HIST_CACHE_TS: dict = {}            # symbol -> datetime when cache was populated
_CK_HIST_CACHE_LOCK = threading.Lock()
_CK_HIST_CACHE_TTL = 3600              # refresh base cache after 1 hour (new session)

# ── ChartInk lag compensation ─────────────────────────────────────────────────
# ChartInk's /oapi endpoint has a known ~5–6 minute data pipeline delay.
# It also includes the *still-forming* current bar in its response, with stale
# OHLCV.  We strip the last N bars from ChartInk and rely on the real-time
# candle builder (Upstox LTP ticks) to supply those bars instead.
#
# CK_BARS_TO_DROP = 2  covers the ~5–6 min lag:
#   bar[-1] = current open bar   → stale partial bar, always drop
#   bar[-2] = last "closed" bar  → often also delayed 1 candle, drop to be safe
# The real-time builder fills these from live LTP ticks automatically.
CK_BARS_TO_DROP  = 2    # Drop this many trailing bars from every CK response
CK_LAG_WARN_MIN  = 7    # Print a warning if last kept CK bar is older than this

# ============ THREAD LOCKS FOR SHARED GLOBALS ============
THREAD_LOCKS = {
    'R3_ALERTED_STOCKS':          threading.RLock(),
    'S3_ALERTED_STOCKS':          threading.RLock(),
    'ACTIVE_POSITIONS': threading.RLock(),
    'DAILY_ORDER_COUNT': threading.RLock(),
    'LAST_ORDER_TIME': threading.RLock(),
    'PLACED_ORDERS': threading.RLock(),
}

# New per-symbol locks for thread safety
CACHE_LOCKS = {}
CACHE_LOCK_MASTER = threading.Lock()

def get_cache_lock(symbol):
    """Get or create a per-symbol lock"""
    with CACHE_LOCK_MASTER:
        if symbol not in CACHE_LOCKS:
            CACHE_LOCKS[symbol] = threading.Lock()
        return CACHE_LOCKS[symbol]

def init_cache_directory():
    """Initialize cache directory structure"""
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)
        print(f"✅ Created cache directory: {CACHE_DIRECTORY}")
    
    # Create subdirectories for organization
    subdirs = ['daily_candles', 'klinger_data', 'metadata']
    for subdir in subdirs:
        path = os.path.join(CACHE_DIRECTORY, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

def get_cache_file_path(symbol, cache_type='daily_candles'):
    """Get cache file path for a symbol"""
    safe_symbol = symbol.replace('|', '_').replace(':', '_')
    return os.path.join(CACHE_DIRECTORY, cache_type, f"{safe_symbol}.csv")

def get_cache_metadata_path(symbol):
    """Get metadata file path for cache entry"""
    safe_symbol = symbol.replace('|', '_').replace(':', '_')
    return os.path.join(CACHE_DIRECTORY, 'metadata', f"{safe_symbol}_meta.json")

def load_candle_cache(symbol, _silent=False):
    """Load cached candles for a symbol.

    Two-level lookup:
      1. In-memory dict  (CANDLE_CACHE)  — zero I/O, no stat increment
      2. Disk CSV + metadata             — increments cache_hits / cache_misses
    Pass _silent=True (used internally by get_cached_or_fetch_candles) to skip
    the debug print on an in-memory hit so we don't log it twice.
    """
    global CACHE_STATS

    if not ENABLE_CANDLE_CACHE:
        return None

    # ── Level 1: in-memory cache (no I/O, no stat bump) ──────────────────────
    if symbol in CANDLE_CACHE:
        df = CANDLE_CACHE[symbol]
        if df is not None and len(df) > 0:
            if DEBUG_MODE and not _silent:
                print(f"✅ Cache hit: {symbol} ({len(df)} candles)")
            return df

    # ── Level 2: disk cache ───────────────────────────────────────────────────
    cache_file = get_cache_file_path(symbol)
    meta_file  = get_cache_metadata_path(symbol)

    if not os.path.exists(cache_file):
        CACHE_STATS['cache_misses'] += 1
        return None

    try:
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            last_update = datetime.fromisoformat(metadata['last_updated'])
            days_old = (datetime.now() - last_update).days
            if days_old > CACHE_EXPIRY_DAYS:
                if DEBUG_MODE:
                    print(f"⚠️ Cache expired for {symbol} ({days_old} days old)")
                CACHE_STATS['cache_misses'] += 1
                return None

        df = pd.read_csv(cache_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        if len(df) > 0:
            CACHE_STATS['cache_hits'] += 1
            # Keep only last 200 candles in RAM — enough for all indicators
            if len(df) > 200:
                df = df.tail(200).reset_index(drop=True)
            CANDLE_CACHE[symbol] = df          # ← populate in-memory layer
            if DEBUG_MODE and not _silent:
                print(f"✅ Cache hit: {symbol} ({len(df)} candles)")
            return df
        else:
            CACHE_STATS['cache_misses'] += 1
            return None

    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Cache load error for {symbol}: {e}")
        CACHE_STATS['cache_misses'] += 1
        return None

def save_candle_cache(symbol, df, instrument_key=None):
    """Save candles to disk cache and refresh in-memory layer"""
    global CACHE_STATS
    
    if not ENABLE_CANDLE_CACHE or df is None or len(df) == 0:
        return False
    
    try:
        cache_file = get_cache_file_path(symbol)
        meta_file = get_cache_metadata_path(symbol)
        
        # Save candles
        df.to_csv(cache_file, index=False)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'instrument_key': instrument_key,
            'last_updated': datetime.now().isoformat(),
            'candle_count': len(df),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            }
        }
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # ── Keep in-memory layer in sync ────────────────────────────────────
        # Trim to last 200 candles — enough for any indicator; saves 50-80% RAM
        if len(df) > 200:
            df = df.tail(200).copy()
        CANDLE_CACHE[symbol] = df.copy()
        
        if DEBUG_MODE:
            print(f"💾 Cached {len(df)} candles for {symbol}")
        
        return True
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Cache save error for {symbol}: {e}")
        return False

def update_candle_cache_incremental(access_token, symbol, instrument_key):
    """Incrementally update cache with only new candles"""
    global CACHE_STATS
    
    # Load existing cache — _silent=True avoids a second "Cache hit" log line
    # when this is called from get_cached_or_fetch_candles after an in-memory hit.
    cached_df = load_candle_cache(symbol, _silent=True)
    
    if cached_df is None or len(cached_df) == 0:
        # No cache - fetch full history
        return fetch_and_cache_full_history(access_token, symbol, instrument_key)
    
    # Get last cached date
    last_cached_date = cached_df['date'].max()
    today = datetime.now().date()
    
    # Check if we need to update
    if last_cached_date.date() >= today:
        if DEBUG_MODE:
            print(f"✅ Cache up-to-date for {symbol}")
        return cached_df
    
    # Fetch only new candles
    from_date = last_cached_date + timedelta(days=1)
    to_date = today
    
    if (to_date - from_date.date()).days < 1:
        return cached_df  # No new data needed
    
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_str}/{from_str}"
        
        resp = _get_upstox_session(access_token).get(url, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            new_candles = data.get("data", {}).get("candles", [])
            
            if new_candles:
                # Convert to DataFrame
                new_df = pd.DataFrame(new_candles, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                new_df['date'] = pd.to_datetime(new_df['date'])
                
                # Merge with existing cache
                merged_df = pd.concat([cached_df, new_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')
                merged_df = merged_df.sort_values('date').reset_index(drop=True)
                
                # Save updated cache
                save_candle_cache(symbol, merged_df, instrument_key)
                
                CACHE_STATS['api_calls_saved'] += 1
                
                if DEBUG_MODE:
                    print(f"📈 Updated cache for {symbol}: +{len(new_candles)} candles")
                
                return merged_df
            else:
                return cached_df
        else:
            if DEBUG_MODE:
                print(f"⚠️ API error updating {symbol}: {resp.status_code}")
            return cached_df
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Incremental update error for {symbol}: {e}")
        return cached_df

def fetch_and_cache_full_history(access_token, symbol, instrument_key, days=120):
    """Fetch full candle history and cache it"""
    global CACHE_STATS
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    from_date_str = start_date.strftime('%Y-%m-%d')
    to_date_str = end_date.strftime('%Y-%m-%d')
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date_str}/{from_date_str}"
    
    try:
        resp = _get_upstox_session(access_token).get(url, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            candles = data.get("data", {}).get("candles", [])
            
            if candles:
                df = pd.DataFrame(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Save to cache
                save_candle_cache(symbol, df, instrument_key)
                
                if DEBUG_MODE:
                    print(f"📥 Fetched and cached {len(df)} candles for {symbol}")
                
                return df
            else:
                return None
        else:
            if DEBUG_MODE:
                print(f"⚠️ API error for {symbol}: {resp.status_code}")
            return None
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Fetch error for {symbol}: {e}")
        return None

def get_cached_or_fetch_candles(access_token, symbol, instrument_key):
    """Get candles from in-memory cache, disk cache, or fetch from API if needed.

    Priority:
      1. In-memory CANDLE_CACHE  — instant, no I/O
      2. Disk CSV cache          — fast, incremental update if stale
      3. Full API fetch          — only when no cache exists at all
    """
    # ── Level 1: in-memory hit (skip disk entirely) ───────────────────────────
    if symbol in CANDLE_CACHE:
        df = CANDLE_CACHE[symbol]
        if df is not None and len(df) >= MIN_CANDLES_FOR_KLINGER:
            last_date = df['date'].max().date()
            today = datetime.now().date()
            if last_date >= today:
                return df          # fully up-to-date, no I/O needed
            # Stale — do incremental update; update_candle_cache_incremental
            # internally calls load_candle_cache(_silent=True) which returns the
            # in-memory copy without re-reading disk or bumping stats again.
            return update_candle_cache_incremental(access_token, symbol, instrument_key)

    # ── Level 2/3: disk cache or full fetch ───────────────────────────────────
    cached_df = load_candle_cache(symbol)          # logs hit/miss once here

    if cached_df is not None and len(cached_df) >= MIN_CANDLES_FOR_KLINGER:
        last_date = cached_df['date'].max().date()
        today = datetime.now().date()
        if last_date < today:
            return update_candle_cache_incremental(access_token, symbol, instrument_key)
        return cached_df
    else:
        return fetch_and_cache_full_history(access_token, symbol, instrument_key)

def calculate_klinger_adaptive(df, symbol=None):
    """Calculate Klinger with adaptive parameters based on data availability"""
    
    if df is None or len(df) < MIN_CANDLES_FOR_KLINGER:
        if DEBUG_MODE and symbol:
            print(f"⚠️ {symbol}: Insufficient data for Klinger ({len(df) if df is not None else 0} candles)")
        return None, None, None

    # Trim to last 200 candles — more than enough for any EMA window;
    # avoids re-processing the entire history on every scan.
    if len(df) > 200:
        df = df.tail(200).reset_index(drop=True)
    
    # Determine which parameters to use
    if ADAPTIVE_KLINGER_LOOKBACK and len(df) < 90:
        # Use shorter parameters for limited data
        fast = KLINGER_FAST_SHORT
        slow = KLINGER_SLOW_SHORT
        signal = KLINGER_SIGNAL_SHORT
        if DEBUG_MODE and symbol:
            print(f"📊 {symbol}: Using adaptive Klinger (fast={fast}, slow={slow}, {len(df)} candles)")
    else:
        # Use standard parameters
        fast = KLINGER_FAST
        slow = KLINGER_SLOW
        signal = KLINGER_SIGNAL
    
    try:
        if len(df) < max(fast, slow, signal) + 10:
            if DEBUG_MODE and symbol:
                print(f"⚠️ {symbol}: Still insufficient data ({len(df)} < {max(fast, slow, signal) + 10})")
            return None, None, None
        
        # Calculate HLC average
        hlc = (df['high'] + df['low'] + df['close']) / 3
        hlc_prev = hlc.shift(1)
        
        # Trend: +1 if up, -1 if down
        trend = ((hlc > hlc_prev).astype(int) * 2 - 1).fillna(0)
        
        # Daily measurement (high - low)
        dm = df['high'] - df['low']
        dm = dm.replace(0, 0.001)  # Avoid division by zero
        
        # Cumulative measurement
        cm = (dm * trend).cumsum()
        cm = cm.replace(0, 0.001).fillna(0.001)  # Avoid division by zero (replace + fillna)
        
        # Volume Force
        volume_force = df['volume'] * trend * (dm / cm) * 100
        # Clip before replacing inf/nan — prevents EWM from inheriting astronomically
        # large values that produce Klinger readings like 1.49e19 (PATANJALI, INDHOTEL etc.)
        volume_force = volume_force.clip(-1e12, 1e12)
        volume_force = volume_force.replace([float('inf'), float('-inf')], 0).fillna(0)
        
        # Apply EMAs to Volume Force
        vf_fast = volume_force.ewm(span=fast, adjust=False).mean()
        vf_slow = volume_force.ewm(span=slow, adjust=False).mean()
        
        # Klinger Oscillator line (clipped to sane range)
        klinger = (vf_fast - vf_slow).clip(-1e12, 1e12)
        
        # Signal line (EMA of Klinger)
        signal_line = klinger.ewm(span=signal, adjust=False).mean()
        
        # Histogram (difference)
        histogram = klinger - signal_line
        
        return klinger, signal_line, histogram
        
    except Exception as e:
        if DEBUG_MODE and symbol:
            print(f"❌ {symbol}: Klinger calculation error: {e}")
        return None, None, None

def fetch_klinger_data_cached(access_token, instrument_key, symbol):
    """Fetch Klinger data using cache system.

    Uses get_cached_or_fetch_candles() which checks the in-memory CANDLE_CACHE
    dict first — so background Klinger refreshes (every 5 scans) do NOT cause
    additional disk reads or bump cache_hits a second time for already-loaded symbols.
    """
    global CACHE_STATS
    
    if not ENABLE_KLINGER_FILTER:
        return None
    
    # get_cached_or_fetch_candles checks in-memory layer first (no I/O, no stat bump)
    df = get_cached_or_fetch_candles(access_token, symbol, instrument_key)
    
    if df is None or len(df) < MIN_CANDLES_FOR_KLINGER:
        candle_count = len(df) if df is not None else 0
        # Always print this — it explains why Klinger shows 0 stocks initialized
        print(f"⚠️ {symbol}: Insufficient candles for Klinger ({candle_count}/{MIN_CANDLES_FOR_KLINGER}) — CE/PE trades blocked for this stock")
        return None
    
    # Calculate Klinger with adaptive parameters
    klinger, signal_line, histogram = calculate_klinger_adaptive(df, symbol)
    
    if klinger is None or len(klinger) < 2:
        if DEBUG_MODE:
            print(f"⚠️ {symbol}: Klinger calculation failed")
        return None
    
    # Return last values for cross detection + history for soft confirmation
    # ko_history: last 5 KO values (oldest first) used for ONGC-type soft confirmation
    ko_history_len = min(5, len(klinger))
    ko_history = [float(klinger.iloc[-(ko_history_len - i)]) for i in range(ko_history_len - 1, -1, -1)]

    return {
        'klinger': float(klinger.iloc[-1]),
        'signal': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1]),
        'klinger_prev': float(klinger.iloc[-2]) if len(klinger) > 1 else float(klinger.iloc[-1]),
        'signal_prev': float(signal_line.iloc[-2]) if len(signal_line) > 1 else float(signal_line.iloc[-1]),
        'ko_history': ko_history,
        'last_update': datetime.now(),
        'candle_count': len(df),
        'adaptive_params': len(df) < 90 if ADAPTIVE_KLINGER_LOOKBACK else False
    }

def cleanup_old_cache():
    """Remove cache files older than expiry period"""
    if not ENABLE_CANDLE_CACHE:
        return
    
    try:
        cleaned_count = 0
        expiry_date = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS * 2)
        
        # Clean daily candles
        candle_dir = os.path.join(CACHE_DIRECTORY, 'daily_candles')
        if os.path.exists(candle_dir):
            for filename in os.listdir(candle_dir):
                filepath = os.path.join(candle_dir, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_mtime < expiry_date:
                    os.remove(filepath)
                    cleaned_count += 1
        
        # Clean metadata
        meta_dir = os.path.join(CACHE_DIRECTORY, 'metadata')
        if os.path.exists(meta_dir):
            for filename in os.listdir(meta_dir):
                filepath = os.path.join(meta_dir, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_mtime < expiry_date:
                    os.remove(filepath)
                    cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} old cache files")
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Cache cleanup error: {e}")

def save_cache_stats():
    """Save cache statistics to file"""
    try:
        CACHE_STATS['last_updated'] = datetime.now().isoformat()
        CACHE_STATS['total_cached_symbols'] = len([f for f in os.listdir(os.path.join(CACHE_DIRECTORY, 'daily_candles')) if f.endswith('.csv')])
        
        stats_file = os.path.join(CACHE_DIRECTORY, CACHE_STATS_FILE)
        with open(stats_file, 'w') as f:
            json.dump(CACHE_STATS, f, indent=2)
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Error saving cache stats: {e}")

def load_cache_stats():
    """Load cache statistics from file"""
    global CACHE_STATS
    
    try:
        stats_file = os.path.join(CACHE_DIRECTORY, CACHE_STATS_FILE)
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                loaded_stats = json.load(f)
                CACHE_STATS.update(loaded_stats)
                
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Error loading cache stats: {e}")

def print_cache_statistics():
    """Print cache performance statistics"""
    if not ENABLE_CANDLE_CACHE:
        return
    
    total_requests = CACHE_STATS['cache_hits'] + CACHE_STATS['cache_misses']
    hit_rate = (CACHE_STATS['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
    mem_symbols = len([s for s, df in CANDLE_CACHE.items() if df is not None and len(df) > 0])
    
    print(f"\n{'='*100}")
    print("💾 CACHE PERFORMANCE STATISTICS")
    print(f"{'='*100}")
    print(f"Cache Hits (disk reads):  {CACHE_STATS['cache_hits']}  ← unique symbols loaded from disk")
    print(f"Cache Misses:             {CACHE_STATS['cache_misses']}  ← symbols fetched fresh from API")
    print(f"Hit Rate:                 {hit_rate:.1f}%")
    print(f"In-Memory Symbols:        {mem_symbols}  ← serving from RAM (no disk I/O)")
    print(f"API Calls Saved:          {CACHE_STATS['api_calls_saved']}")
    print(f"Cached Symbols (disk):    {CACHE_STATS['total_cached_symbols']}")
    print(f"Cache Directory:          {CACHE_DIRECTORY}")
    print(f"{'='*100}\n")

# ============================================================================
# ORIGINAL FUNCTIONS - UPDATED TO USE CACHE
# ============================================================================

# Replace the original calculate_klinger with the adaptive version
def calculate_klinger(df, fast=34, slow=55, signal=13):
    """
    DEPRECATED: Use calculate_klinger_adaptive instead
    This is kept for backward compatibility
    """
    return calculate_klinger_adaptive(df)

# Replace the original fetch_klinger_data with cached version
def fetch_klinger_data(access_token, instrument_key, days=90):
    """
    Fetch Klinger data using cache (backward compatible wrapper)
    """
    # Extract symbol from instrument_key
    symbol = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key.split(':')[-1]
    return fetch_klinger_data_cached(access_token, instrument_key, symbol)

# ============ UPSTOX TOKEN GENERATION CLASS ============
class UpstoxLogin:
    # ... (unchanged from original both4) ...
    def __init__(self, mobile_number: str,
                 email_address: str,
                 email_password: str,
                 passcode: str = None,
                 cookies_file: str = "upstox_cookies.pkl",
                 login_url: str = "https://login.upstox.com",
                 target_url: str = "https://account.upstox.com/developer/apps",
                 max_retries: int = 3):
        self.mobile_number = mobile_number
        self.email_address = email_address
        self.email_password = email_password
        self.passcode = passcode
        self.login_url = login_url
        self.target_url = target_url
        self.cookies_file = cookies_file
        self.driver = None
        self.max_retries = max_retries

    # ... (rest of the class remains unchanged) ...
    # For brevity, the class is exactly as in the original both4.
    # Please keep all methods unchanged.
    def check_token_timestamp(self) -> bool:
        try:
            if not os.path.exists(TOKEN_TIMESTAMP_FILE):
                print("ℹ️ No token timestamp found")
                return False
            with open(TOKEN_TIMESTAMP_FILE, 'r') as f:
                data = json.load(f)
                token_timestamp = datetime.fromisoformat(data['timestamp'])
                token_date = token_timestamp.date()
            now = datetime.now()
            today = now.date()
            cutoff_time = datetime.combine(today, datetime.strptime("09:00", "%H:%M").time())
            print(f"📅 Token generated on: {token_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📅 Today's date: {today}")
            print(f"⏰ Current time: {now.strftime('%H:%M:%S')}")
            print(f"⏰ Cutoff time: 09:00 AM")
            if token_date == today and token_timestamp < cutoff_time:
                print("✅ Token already generated today before 9:00 AM")
                print("✅ No need to regenerate - exiting script")
                return True
            elif token_date == today and token_timestamp >= cutoff_time:
                print("⚠️ Token generated today after 9:00 AM - will regenerate")
                return False
            else:
                print("⚠️ Token is from a previous date - will regenerate")
                return False
        except Exception as e:
            print(f"⚠️ Error reading token timestamp: {e}")
            return False

    def save_token_timestamp(self, token: str):
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'token': token,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S')
            }
            with open(TOKEN_TIMESTAMP_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            with open(UPSTOX_TOKEN_FILE, 'w') as f:
                f.write(token)
            print(f"✅ Token timestamp saved: {data['date']} {data['time']}")
            print(f"📁 Token saved to: {UPSTOX_TOKEN_FILE}")
        except Exception as e:
            print(f"⚠️ Error saving token timestamp: {e}")

    def setup_driver(self, headless: bool = False):
        chrome_options = webdriver.ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,800")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    # ── HEADLESS OAUTH FLOW ───────────────────────────────────────────────────
    def perform_oauth_headless(self) -> str:
        """
        Fully automated headless Selenium OAuth flow.

        Spins up a local HTTP server on HEADLESS_SERVER_PORT to capture the
        redirect code, then drives Chrome headlessly through:
          mobile number → OTP (via email) → PIN → consent → token exchange.

        No manual browser interaction needed — works on servers, GitHub Actions,
        and headless Linux/Windows environments.

        Returns the access_token string, or None on failure.
        """
        import urllib.parse as _urlparse

        # ── 1. Local redirect-capture server ─────────────────────────────────
        _captured_code = [None]
        _server_ready  = threading.Event()

        class _CodeHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                code   = parse_qs(parsed.query).get("code", [None])[0]
                if code:
                    _captured_code[0] = code
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<h1>Auth successful - you may close this tab.</h1>")
                    threading.Thread(target=lambda: (time.sleep(1), self.server.shutdown()), daemon=True).start()
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing code")
            def log_message(self, *_):
                pass

        def _run_server():
            try:
                srv = HTTPServer(("127.0.0.1", HEADLESS_SERVER_PORT), _CodeHandler)
                _server_ready.set()
                print(f"✅ Headless redirect server on http://127.0.0.1:{HEADLESS_SERVER_PORT}")
                srv.serve_forever()
            except OSError as exc:
                print(f"❌ Cannot bind port {HEADLESS_SERVER_PORT}: {exc}")
                _server_ready.set()

        threading.Thread(target=_run_server, daemon=True).start()
        if not _server_ready.wait(timeout=5):
            print("❌ Redirect server failed to start.")
            return None

        # ── 2. Headless Chrome login ──────────────────────────────────────────
        print("\n🤖 HEADLESS OAUTH LOGIN")
        self.setup_driver(headless=True)
        try:
            auth_url = (
                "https://api.upstox.com/v2/login/authorization/dialog"
                f"?response_type=code"
                f"&client_id={UPSTOX_API_KEY}"
                f"&redirect_uri={_urlparse.quote(UPSTOX_REDIRECT_URI, safe='')}"
            )
            print("🌐 Opening login page (headless)…")
            self.driver.get(auth_url)
            time.sleep(2)

            # Clear old emails before requesting OTP
            self.delete_all_upstox_emails()

            # Mobile number
            mobile_input = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "mobileNum"))
            )
            mobile_input.clear()
            mobile_input.send_keys(self.mobile_number)

            # Click "Get OTP"
            get_otp_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Get OTP')]"))
            )
            otp_request_time = datetime.now()
            get_otp_btn.click()
            print("📨 OTP requested.")

            # Fetch OTP from email (reuses existing method)
            otp = self.get_latest_otp_by_uid(max_wait=120, otp_request_time=otp_request_time)
            if not otp:
                print("❌ OTP not retrieved.")
                return None

            # Enter OTP
            otp_input = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "otpNum"))
            )
            for digit in str(otp):
                otp_input.send_keys(digit)
                time.sleep(0.1)

            # Continue / Verify
            continue_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'Continue') or contains(text(), 'Verify')]")
                )
            )
            continue_btn.click()
            print("✅ OTP submitted.")
            time.sleep(3)

            # PIN (optional — skipped gracefully if not present)
            if self.passcode:
                try:
                    pin_input = WebDriverWait(self.driver, 8).until(
                        EC.visibility_of_element_located((By.ID, "pinCode"))
                    )
                    pin_input.clear()
                    pin_input.send_keys(self.passcode)
                    try:
                        pin_continue = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.ID, "pinContinueBtn"))
                        )
                        pin_continue.click()
                    except Exception:
                        # Fall back to generic submit
                        for sel in [(By.XPATH, "//button[@type='submit']"),
                                    (By.XPATH, "//button[contains(text(), 'Continue')]")]:
                            try:
                                WebDriverWait(self.driver, 3).until(
                                    EC.element_to_be_clickable(sel)
                                ).click()
                                break
                            except Exception:
                                continue
                    print("✅ PIN submitted.")
                    time.sleep(3)
                except Exception as e:
                    print(f"ℹ️ PIN step skipped: {e}")

            # Consent / Allow button (some Upstox flows show this)
            try:
                allow_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Allow')]"))
                )
                allow_btn.click()
                print("✅ Consent granted.")
                time.sleep(2)
            except Exception:
                pass  # Not always shown

            # ── 3. Wait for redirect and capture auth code ────────────────────
            print("⏳ Waiting for OAuth redirect…")
            deadline = time.time() + 30
            while time.time() < deadline:
                cur = self.driver.current_url
                if "127.0.0.1" in cur:
                    parsed = urlparse(cur)
                    code   = parse_qs(parsed.query).get("code", [None])[0]
                    if code:
                        _captured_code[0] = code
                        print("✅ Auth code captured from browser URL.")
                        break
                if _captured_code[0]:
                    break
                time.sleep(1)

            if not _captured_code[0]:
                print("❌ No auth code received after 30 s.")
                return None

        except Exception as exc:
            print(f"❌ Headless login error: {exc}")
            return None
        finally:
            self.close()

        # ── 4. Exchange code for access token ─────────────────────────────────
        print("🔄 Exchanging auth code for access token…")
        try:
            resp = requests.post(
                "https://api.upstox.com/v2/login/authorization/token",
                headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "code":          _captured_code[0],
                    "client_id":     UPSTOX_API_KEY,
                    "client_secret": UPSTOX_API_SECRET,
                    "redirect_uri":  UPSTOX_REDIRECT_URI,
                    "grant_type":    "authorization_code",
                },
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"❌ Token exchange failed ({resp.status_code}): {resp.text[:200]}")
                return None
            data          = resp.json()
            access_token  = data.get("access_token")
            refresh_token = data.get("refresh_token", "")
            if not access_token:
                print(f"❌ No access_token in response: {resp.text[:200]}")
                return None

            # Save refresh token for future runs
            try:
                with open(UPSTOX_REFRESH_TOKEN_FILE, "w") as rf:
                    rf.write(refresh_token)
                print(f"💾 Refresh token saved to {UPSTOX_REFRESH_TOKEN_FILE}")
            except Exception:
                pass

            print("✅ Headless OAuth complete!")
            return access_token
        except Exception as exc:
            print(f"❌ Token exchange error: {exc}")
            return None
    # ── END HEADLESS OAUTH FLOW ───────────────────────────────────────────────

    def load_cookies(self) -> bool:
        if not os.path.exists(self.cookies_file):
            return False
        try:
            self.driver.get(self.login_url)
            time.sleep(2)
            cookies = pickle.load(open(self.cookies_file, "rb"))
            for c in cookies:
                self.driver.add_cookie(c)
            print("✅ Cookies loaded")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load cookies: {e}")
            return False

    def save_cookies(self):
        try:
            pickle.dump(self.driver.get_cookies(), open(self.cookies_file, "wb"))
            print("✅ Cookies saved")
        except Exception as e:
            print(f"⚠️ Failed to save cookies: {e}")

    def delete_all_upstox_emails(self):
        try:
            with imaplib.IMAP4_SSL("imap.gmail.com") as mail:
                mail.login(self.email_address, self.email_password)
                mail.select("inbox")
                search_patterns = [
                    'FROM "donotreply@transactions.upstox.com"',
                    'FROM "upstox"'
                ]
                deleted_count = 0
                for pattern in search_patterns:
                    try:
                        status, messages = mail.search(None, pattern)
                        if status == "OK" and messages[0]:
                            email_ids = messages[0].split()
                            for email_id in email_ids:
                                mail.store(email_id, '+FLAGS', '\\Deleted')
                                deleted_count += 1
                            mail.expunge()
                    except:
                        pass
                print(f"✅ Deleted {deleted_count} Upstox emails from inbox")
        except Exception as e:
            print(f"⚠️ Error deleting emails: {e}")

    def get_latest_otp_by_uid(self, max_wait: int = 90, otp_request_time: datetime = None) -> str:
        try:
            if otp_request_time is None:
                otp_request_time = datetime.now()
            print(f"⏳ Waiting for NEW OTP email (checking every 3 seconds, max {max_wait}s)...")
            print(f"🕐 OTP requested at: {otp_request_time.strftime('%H:%M:%S')}")
            start_time = time.time()
            check_interval = 3
            print("⏱️ Waiting 8 seconds for email to arrive...")
            time.sleep(8)
            while time.time() - start_time < max_wait:
                try:
                    with imaplib.IMAP4_SSL("imap.gmail.com") as mail:
                        mail.login(self.email_address, self.email_password)
                        mail.select("inbox")
                        status, messages = mail.search(None, '(UNSEEN FROM "donotreply@transactions.upstox.com")')
                        if status != "OK" or not messages[0]:
                            # Fallback: Upstox sometimes sends from a different subdomain
                            status, messages = mail.search(None, '(UNSEEN FROM "upstox")')
                        if status != "OK" or not messages[0]:
                            elapsed = int(time.time() - start_time)
                            print(f"⏳ No unread Upstox emails yet... ({elapsed}s)")
                            time.sleep(check_interval)
                            continue
                        email_ids = messages[0].split()
                        print(f"📬 Found {len(email_ids)} unread email(s) from Upstox")
                        for email_id in reversed(email_ids):
                            try:
                                status, msg_data = mail.fetch(email_id, "(RFC822 INTERNALDATE)")
                                if status != "OK":
                                    continue
                                internaldate_pattern = rb'INTERNALDATE "([^"]+)"'
                                internaldate_match = re.search(internaldate_pattern, msg_data[0][0])
                                if internaldate_match:
                                    internaldate_str = internaldate_match.group(1).decode()
                                    try:
                                        email_received_time = datetime.strptime(internaldate_str, "%d-%b-%Y %H:%M:%S %z")
                                        email_received_time = email_received_time.replace(tzinfo=None)
                                        time_diff = (email_received_time - otp_request_time).total_seconds()
                                        if time_diff < -5:
                                            print(f"⏭️ Skipping old email (received {abs(int(time_diff))}s BEFORE request)")
                                            continue
                                        else:
                                            print(f"✅ Found FRESH email (received {int(time_diff) if time_diff > 0 else 0}s after request)")
                                    except Exception as e:
                                        print(f"⚠️ Could not parse INTERNALDATE: {e}")
                                        continue
                                msg = email.message_from_bytes(msg_data[0][1])
                                sender = msg.get("From", "")
                                subject = msg.get("Subject", "")
                                body = ""
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain":
                                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                            break
                                        elif part.get_content_type() == "text/html":
                                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                else:
                                    body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
                                body = body.replace("\r", " ").replace("\n", " ").replace("\t", " ")
                                print(f"📄 Email body preview: {body[:200]}...")
                                otp_patterns = [
                                    r'OTP\s*(?:is)?\s*[:=]?\s*(\d{6})',
                                    r'one.?time.?password\s*(?:is)?\s*[:=]?\s*(\d{6})',
                                    r'verification.?code\s*(?:is)?\s*[:=]?\s*(\d{6})',
                                    r'code\s*(?:is)?\s*[:=]?\s*(\d{6})',
                                    r'(?:<b>|>)\s*(\d{6})\s*(?:</b>|<)',
                                    r'\b(\d{6})\b(?!.*\d{7,})',
                                ]
                                found_otp = None
                                for pattern in otp_patterns:
                                    otp_match = re.search(pattern, body, re.IGNORECASE)
                                    if otp_match:
                                        found_otp = otp_match.group(1)
                                        print(f"🎯 Pattern '{pattern}' matched OTP: {found_otp}")
                                        break
                                if not found_otp:
                                    all_numbers = re.findall(r'\b\d{6}\b', body)
                                    print(f"🔍 Found {len(all_numbers)} potential 6-digit numbers in email")
                                    for num in all_numbers:
                                        if len(num) == 6:
                                            context_start = max(body.find(num) - 20, 0)
                                            context_end = min(body.find(num) + 26, len(body))
                                            context = body[context_start:context_end].lower()
                                            otp_keywords = ['otp', 'one time password', 'verification code', 'code', 'password']
                                            if any(keyword in context for keyword in otp_keywords):
                                                found_otp = num
                                                print(f"🎯 Found OTP by context: {found_otp}")
                                                break
                                            elif context_start == 0:
                                                found_otp = num
                                                print(f"🎯 Found OTP at beginning: {found_otp}")
                                                break
                                if found_otp:
                                    print(f"\n✅ OTP EXTRACTED: {found_otp}")
                                    print(f"   📧 From: {sender}")
                                    print(f"   📋 Subject: {subject}")
                                    print(f"   ⏰ Received: {email_received_time.strftime('%H:%M:%S') if 'email_received_time' in locals() else 'N/A'}")
                                    mail.store(email_id, '+FLAGS', '\\Seen')
                                    mail.store(email_id, '+FLAGS', '\\Deleted')
                                    mail.expunge()
                                    return found_otp
                                else:
                                    print("❌ Could not extract OTP from email body")
                            except Exception as e:
                                print(f"⚠️ Error processing email: {e}")
                                continue
                except Exception as e:
                    print(f"⚠️ IMAP check failed: {e}")
                elapsed = int(time.time() - start_time)
                print(f"⏳ Still waiting for OTP... ({elapsed}s elapsed)")
                time.sleep(check_interval)
            print(f"❌ No OTP received after {max_wait} seconds")
            return None
        except Exception as e:
            print(f"❌ Error in OTP retrieval: {e}")
            return None

    def wait_for_cloudflare_checkbox(self, timeout: int = 20):
        print("🔍 Checking for Cloudflare verification...")
        selectors = [
            (By.CSS_SELECTOR, "input[type='checkbox']"),
            (By.XPATH, "//input[@type='checkbox']"),
            (By.XPATH, "//iframe[contains(@src, 'cloudflare')]"),
        ]
        for selector_type, selector_value in selectors:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((selector_type, selector_value))
                )
                if selector_type == By.XPATH and "iframe" in selector_value:
                    self.driver.switch_to.frame(element)
                    checkbox = self.driver.find_element(By.CSS_SELECTOR, "input[type='checkbox']")
                    checkbox.click()
                    self.driver.switch_to.default_content()
                else:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                    time.sleep(0.5)
                    try:
                        element.click()
                    except:
                        self.driver.execute_script("arguments[0].click();", element)
                print("✅ Cloudflare verification handled")
                time.sleep(2)
                return True
            except TimeoutException:
                continue
            except Exception:
                continue
        print("ℹ️ No Cloudflare checkbox found")
        return False

    def revoke_token(self, app_name: str = "Feroz") -> bool:
        try:
            print(f"\n🔄 STEP 12: Revoking old token for '{app_name}'...")
            time.sleep(2)
            app_rows = self.driver.find_elements(By.XPATH, "//table//tbody//tr")
            print(f"📋 Found {len(app_rows)} app(s) in table")
            target_row = None
            for row in app_rows:
                try:
                    if app_name.lower() in row.text.lower():
                        target_row = row
                        print(f"✅ Found app row for '{app_name}'")
                        break
                except:
                    continue
            if not target_row:
                print(f"⚠️ App '{app_name}' not found, trying first connected app...")
                for row in app_rows:
                    try:
                        if "connected" in row.text.lower():
                            target_row = row
                            print(f"✅ Found connected app row")
                            break
                    except:
                        continue
            if not target_row:
                print("❌ No suitable app found")
                return False
            dropdown_buttons = target_row.find_elements(By.XPATH, ".//button")
            dropdown_clicked = False
            for btn in dropdown_buttons:
                try:
                    btn_html = btn.get_attribute('outerHTML')
                    if any(indicator in btn_html.lower() for indicator in ['chevron', 'arrow', 'expand', 'dropdown']):
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                        time.sleep(0.5)
                        btn.click()
                        print("✅ Clicked dropdown arrow")
                        dropdown_clicked = True
                        time.sleep(2)
                        break
                except:
                    continue
            if not dropdown_clicked and len(dropdown_buttons) > 0:
                try:
                    last_btn = dropdown_buttons[-1]
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", last_btn)
                    time.sleep(0.5)
                    last_btn.click()
                    print("✅ Clicked dropdown button")
                    time.sleep(2)
                except Exception as e:
                    print(f"⚠️ Could not click dropdown: {e}")
            try:
                revoke_selectors = [
                    (By.XPATH, "//span[contains(text(), 'Revoke')]"),
                    (By.XPATH, "//button[contains(., 'Revoke')]"),
                    (By.CSS_SELECTOR, "span.dj.az.dl.ce.bc"),
                ]
                revoke_button = None
                for selector_type, selector_value in revoke_selectors:
                    try:
                        revoke_button = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((selector_type, selector_value))
                        )
                        break
                    except:
                        continue
                if revoke_button:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", revoke_button)
                    time.sleep(0.5)
                    try:
                        revoke_button.click()
                    except:
                        parent_btn = revoke_button.find_element(By.XPATH, "./..")
                        parent_btn.click()
                    print("✅ Clicked Revoke button")
                    time.sleep(2)
                    print("⏳ Waiting for confirmation modal...")
                    try:
                        confirm_selectors = [
                            (By.XPATH, "//button[contains(., 'Confirm')]"),
                            (By.XPATH, "/html/body/div/div/div[2]/div[2]/div/button[2]"),
                            (By.CSS_SELECTOR, "button.cd.az.ba.dn.cf.a.ey.do.ez.av.c.dp.d.dq.dr.ds.fb.fc.fd.fe.ff.fg.fh.ea.fi.fj.fk"),
                        ]
                        confirm_button = None
                        for selector_type, selector_value in confirm_selectors:
                            try:
                                confirm_button = WebDriverWait(self.driver, 10).until(
                                    EC.element_to_be_clickable((selector_type, selector_value))
                                )
                                print(f"✅ Found Confirm button")
                                break
                            except:
                                continue
                        if confirm_button:
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", confirm_button)
                            time.sleep(0.5)
                            try:
                                confirm_button.click()
                                print("✅ Clicked Confirm button (direct)")
                            except ElementClickInterceptedException:
                                self.driver.execute_script("arguments[0].click();", confirm_button)
                                print("✅ Clicked Confirm button (JavaScript)")
                            time.sleep(3)
                            print("⏳ Waiting for modal to close...")
                            time.sleep(2)
                            print("✅ Token revoked successfully")
                            return True
                        else:
                            print("⚠️ Confirm button not found")
                            return False
                    except Exception as e:
                        print(f"❌ Error handling confirmation modal: {e}")
                        return False
                else:
                    print("⚠️ Revoke button not found - token may already be revoked or not exist")
                    return False
            except Exception as e:
                print(f"❌ Error clicking Revoke button: {e}")
                return False
        except Exception as e:
            print(f"❌ Error revoking token: {e}")
            return False

    def generate_token(self, app_name: str = "Feroz") -> bool:
        try:
            print(f"\n🔄 STEP 12B: Generating new token for '{app_name}'...")
            time.sleep(3)
            generate_selectors = [
                (By.XPATH, "//button[contains(., 'Generate')]"),
                (By.XPATH, "//span[contains(text(), 'Generate')]"),
                (By.XPATH, "//button[.//span[contains(text(), 'Generate')]]"),
                (By.CSS_SELECTOR, "button[type='button']"),
                (By.XPATH, "//div[contains(text(), 'Generate')]"),
                (By.XPATH, "//*[contains(text(), 'Generate Token')]"),
            ]
            generate_button = None
            max_attempts = 3
            for attempt in range(max_attempts):
                print(f"🔍 Attempt {attempt + 1}/{max_attempts} to find Generate button...")
                for selector_type, selector_value in generate_selectors:
                    try:
                        elements = self.driver.find_elements(selector_type, selector_value)
                        for element in elements:
                            try:
                                element_text = element.text.strip().lower()
                                if "generate" in element_text and "revoke" not in element_text:
                                    generate_button = element
                                    print(f"✅ Found Generate button with text: {element.text}")
                                    break
                            except:
                                continue
                        if generate_button:
                            break
                    except:
                        continue
                if generate_button:
                    break
                if attempt < max_attempts - 1:
                    print("⚠️ Generate button not found, checking dropdowns...")
                    dropdown_arrows = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'chevron') or contains(@aria-label, 'expand') or .//*[contains(text(), '▼') or contains(text(), '▾')]]")
                    for arrow in dropdown_arrows[:3]:
                        try:
                            arrow.click()
                            print("✅ Clicked dropdown arrow")
                            time.sleep(2)
                            break
                        except:
                            continue
                    time.sleep(2)
            if not generate_button:
                print("ℹ️ Generate button not found, checking if token already exists...")
                token_indicators = [
                    "eyJ",
                    "access token",
                    "bearer",
                    "token:"
                ]
                page_source = self.driver.page_source.lower()
                existing_token_detected = any(indicator.lower() in page_source for indicator in token_indicators)
                if existing_token_detected:
                    print("✅ Token appears to already exist")
                    return True
                else:
                    print("⚠️ No Generate button found and no existing token detected")
                    return False
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", generate_button)
            time.sleep(1)
            click_methods = [
                ("direct click", lambda: generate_button.click()),
                ("JavaScript click", lambda: self.driver.execute_script("arguments[0].click();", generate_button)),
                ("parent click", lambda: generate_button.find_element(By.XPATH, "./..").click()),
            ]
            click_success = False
            for method_name, click_func in click_methods:
                try:
                    click_func()
                    print(f"✅ Clicked Generate button ({method_name})")
                    click_success = True
                    break
                except Exception as e:
                    print(f"⚠️ {method_name} failed: {e}")
                    continue
            if not click_success:
                print("❌ All click methods failed")
                return False
            print("⏳ Waiting for confirmation modal to appear...")
            time.sleep(3)
            print("🔍 Looking for confirmation modal...")
            try:
                modal_selectors = [
                    (By.XPATH, "//*[contains(text(), 'Confirm generate?')]"),
                    (By.XPATH, "//*[contains(text(), 'Are you sure you want to generate')]"),
                    (By.XPATH, "//div[contains(@class, 'modal')]"),
                    (By.XPATH, "//div[@role='dialog']"),
                    (By.CSS_SELECTOR, "div[class*='modal']"),
                ]
                modal_found = False
                for selector_type, selector_value in modal_selectors:
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((selector_type, selector_value))
                        )
                        print(f"✅ Confirmation modal found")
                        modal_found = True
                        break
                    except:
                        continue
                if not modal_found:
                    print("⚠️ No confirmation modal found, proceeding...")
                else:
                    print("🔍 Looking for Confirm button in modal...")
                    confirm_selectors = [
                        (By.XPATH, "//button[contains(., 'Confirm') and not(contains(., 'Cancel'))]"),
                        (By.XPATH, "//button[text()='Confirm']"),
                        (By.XPATH, "//button[contains(text(), 'Confirm')]"),
                        (By.XPATH, "//button[@type='button' and contains(., 'Confirm')]"),
                        (By.CSS_SELECTOR, "button[class*='confirm']"),
                    ]
                    confirm_button = None
                    for selector_type, selector_value in confirm_selectors:
                        try:
                            confirm_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((selector_type, selector_value))
                            )
                            button_text = confirm_button.text.strip().lower()
                            if "cancel" not in button_text:
                                print(f"✅ Found Confirm button with text: {confirm_button.text}")
                                break
                            else:
                                confirm_button = None
                        except:
                            continue
                    if confirm_button:
                        self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", confirm_button)
                        time.sleep(1)
                        try:
                            confirm_button.click()
                            print("✅ Clicked Confirm button in modal")
                        except ElementClickInterceptedException:
                            self.driver.execute_script("arguments[0].click();", confirm_button)
                            print("✅ Clicked Confirm button (JavaScript)")
                        print("⏳ Waiting for token generation (10 seconds)...")
                        time.sleep(10)
                    else:
                        print("⚠️ Confirm button not found in modal")
                        print("⏳ Waiting for token generation anyway (10 seconds)...")
                        time.sleep(10)
            except Exception as e:
                print(f"⚠️ Error handling confirmation modal: {e}")
                print("⏳ Continuing with token generation (10 seconds)...")
                time.sleep(10)
            try:
                success_indicators = [
                    "Token generated successfully",
                    "successfully generated",
                    "access token",
                    "eyJ",
                ]
                page_source = self.driver.page_source.lower()
                if any(indicator.lower() in page_source for indicator in success_indicators):
                    print("✅ Token generation successful")
                    return True
                else:
                    error_indicators = ["error", "failed", "unable", "cannot"]
                    if any(indicator in page_source for indicator in error_indicators):
                        print("⚠️ Token generation may have failed - error detected")
                    else:
                        print("ℹ️ Token generation status unclear, but proceeding")
                    return True
            except Exception as e:
                print(f"⚠️ Error verifying token generation: {e}")
                return True
        except Exception as e:
            print(f"❌ Error generating token: {e}")
            import traceback
            traceback.print_exc()
            return False

    def copy_access_token(self, app_name: str = "Feroz") -> str:
        try:
            print(f"\n🔑 STEP 13: Copying new access token for '{app_name}'...")
            time.sleep(5)
            token = self._copy_access_token_specific()
            if token and len(token) > 100 and token.startswith('eyJ'):
                print(f"✅ Access Token found: {token[:20]}...{token[-10:]}")
                return token
            strategies = [
                self._copy_from_jwt_dom_elements,
                self._copy_from_token_field,
                self._copy_from_api_key_section,
            ]
            for i, strategy in enumerate(strategies):
                try:
                    print(f"🔍 Trying fallback strategy {i+1}/{len(strategies)}...")
                    token = strategy()
                    if token and len(token) > 100 and token.startswith('eyJ'):
                        print(f"✅ Access Token found using fallback {i+1}: {token[:20]}...{token[-10:]}")
                        return token
                except Exception as e:
                    print(f"⚠️ Strategy {i+1} failed: {e}")
                    continue
            print("❌ Could not retrieve Access Token using any method")
            return None
        except Exception as e:
            print(f"❌ Error copying token: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _copy_access_token_specific(self) -> str:
        try:
            access_token_labels = self.driver.find_elements(
                By.XPATH, 
                "//*[contains(text(), 'Access Token') or contains(text(), 'access_token') or contains(text(), 'Bearer')]"
            )
            print(f"🔍 Found {len(access_token_labels)} elements mentioning Access Token")
            for label in access_token_labels:
                try:
                    container = label.find_element(By.XPATH, "./ancestor::div[1]")
                    copy_buttons = container.find_elements(
                        By.XPATH, 
                        ".//button[.//img[@alt='copy']] | .//button[contains(@class, 'copy')] | .//img[@alt='copy']"
                    )
                    print(f"📋 Found {len(copy_buttons)} copy buttons near Access Token label")
                    for i, copy_btn in enumerate(copy_buttons):
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", copy_btn)
                            time.sleep(0.5)
                            print(f"🖱️ Clicking copy button #{i+1} near Access Token...")
                            try:
                                copy_btn.click()
                            except:
                                parent = copy_btn.find_element(By.XPATH, "./..")
                                parent.click()
                            time.sleep(1)
                            token = pyperclip.paste()
                            if token and len(token) > 100 and token.startswith('eyJ'):
                                print(f"✅ JWT Access Token found: {token[:20]}...")
                                return token
                            else:
                                if token:
                                    print(f"⚠️ Copied value is NOT a JWT token (length: {len(token)}): {token[:30] if len(token) > 30 else token}")
                                token_text = self._find_jwt_nearby(container)
                                if token_text and len(token_text) > 100 and token_text.startswith('eyJ'):
                                    print(f"✅ Found JWT token in nearby text: {token_text[:20]}...")
                                    return token_text
                        except Exception as e:
                            print(f"⚠️ Copy button #{i+1} failed: {e}")
                            continue
                except Exception as e:
                    print(f"⚠️ Container search failed: {e}")
                    continue
            print("🔍 Searching entire page for JWT tokens...")
            all_elements = self.driver.find_elements(By.XPATH, "//*")
            for element in all_elements:
                try:
                    text = element.text
                    if text and 'eyJ' in text and len(text) > 100:
                        import re
                        token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', text)
                        if token_match:
                            token = token_match.group(1)
                            print(f"✅ Found JWT token in page text: {token[:20]}...")
                            return token
                except:
                    continue
            return None
        except Exception as e:
            print(f"⚠️ Specific Access Token search failed: {e}")
            return None

    def _find_jwt_nearby(self, container):
        try:
            container_text = container.text
            import re
            token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', container_text)
            if token_match:
                return token_match.group(1)
            try:
                parent = container.find_element(By.XPATH, "./..")
                parent_text = parent.text
                token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', parent_text)
                if token_match:
                    return token_match.group(1)
            except:
                pass
        except Exception as e:
            print(f"⚠️ JWT search failed: {e}")
        return None

    def _copy_from_jwt_dom_elements(self) -> str:
        try:
            all_elements = self.driver.find_elements(By.XPATH, "//*")
            for element in all_elements:
                try:
                    text = element.text
                    if text and 'eyJ' in text and len(text) > 100:
                        import re
                        token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', text)
                        if token_match:
                            token = token_match.group(1)
                            print(f"✅ Found JWT token in DOM: {token[:20]}...")
                            return token
                except:
                    continue
            page_source = self.driver.page_source
            import re
            token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', page_source)
            if token_match:
                token = token_match.group(1)
                print(f"✅ Found JWT token in page source: {token[:20]}...")
                return token
            return None
        except Exception as e:
            print(f"⚠️ JWT DOM strategy failed: {e}")
            return None

    def _copy_from_token_field(self) -> str:
        try:
            input_fields = self.driver.find_elements(By.XPATH, "//input[@type='text' or @type='password' or @type='hidden']")
            for field in input_fields:
                try:
                    value = field.get_attribute('value')
                    if value and len(value) > 100 and value.startswith('eyJ'):
                        print(f"✅ Found token in input field: {value[:20]}...")
                        return value
                except:
                    continue
            return None
        except Exception as e:
            print(f"⚠️ Token field strategy failed: {e}")
            return None

    def _copy_from_api_key_section(self) -> str:
        try:
            sections = self.driver.find_elements(By.XPATH, "//div[contains(text(), 'API') or contains(text(), 'Token')]")
            for section in sections:
                try:
                    container = section.find_element(By.XPATH, "./ancestor::div[1]")
                    container_text = container.text
                    token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', container_text)
                    if token_match:
                        token = token_match.group(1)
                        print(f"✅ Found token in API section: {token[:20]}...")
                        return token
                except:
                    continue
            page_source = self.driver.page_source
            token_match = re.search(r'(eyJ[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+?\.[a-zA-Z0-9\-_]+)', page_source)
            if token_match:
                token = token_match.group(1)
                print(f"✅ Found token in page source: {token[:20]}...")
                return token
            return None
        except Exception as e:
            print(f"⚠️ API section strategy failed: {e}")
            return None

    def login_attempt(self) -> bool:
        print("🌐 Opening Upstox login page...")
        self.driver.get(self.login_url)
        time.sleep(3)
        print("\n📧 STEP 1: Clearing all Upstox emails...")
        self.delete_all_upstox_emails()
        time.sleep(2)
        try:
            print("\n📱 STEP 2: Entering mobile number...")
            mobile_input = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "mobileNum"))
            )
            mobile_input.clear()
            mobile_input.send_keys(self.mobile_number)
            time.sleep(1)
            print("✅ Mobile number entered")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
        print("\n🛡️ STEP 3: Checking Cloudflare...")
        self.wait_for_cloudflare_checkbox()
        time.sleep(2)
        try:
            print("\n📘 STEP 4: Requesting OTP...")
            get_otp_button = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Get OTP')]"))
            )
            otp_request_time = datetime.now()
            print(f"🕐 Timestamp: {otp_request_time.strftime('%H:%M:%S.%f')[:-3]}")
            get_otp_button.click()
            print("✅ OTP requested")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
        try:
            print("\n⏳ STEP 5: Waiting for OTP input field...")
            otp_input = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "otpNum"))
            )
            print("✅ OTP field ready")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
        print("\n📬 STEP 6: Retrieving OTP from email...")
        otp = self.get_latest_otp_by_uid(max_wait=90, otp_request_time=otp_request_time)
        if not otp:
            print("❌ No OTP received")
            return False
        try:
            print(f"\n📝 STEP 7: Entering OTP...")
            otp_input.clear()
            for char in str(otp):
                otp_input.send_keys(char)
                time.sleep(0.1)
            time.sleep(1)
            print(f"✅ Entered: {otp}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
        try:
            print("\n📘 STEP 8: Submitting OTP...")
            verify_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue') or contains(text(), 'Verify')]"))
            )
            verify_button.click()
            print("✅ Submitted")
            time.sleep(4)
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
        if self.passcode:
            try:
                print("\n🔐 STEP 9: Entering passcode...")
                passcode_selectors = [
                    (By.ID, "pinCode"),
                    (By.XPATH, "//input[@type='password']"),
                    (By.XPATH, "//input[@placeholder='Enter PIN']"),
                    (By.CSS_SELECTOR, "input[type='password']")
                ]
                passcode_input = None
                for selector_type, selector_value in passcode_selectors:
                    try:
                        passcode_input = WebDriverWait(self.driver, 5).until(
                            EC.visibility_of_element_located((selector_type, selector_value))
                        )
                        break
                    except:
                        continue
                if passcode_input:
                    passcode_input.clear()
                    passcode_input.send_keys(self.passcode)
                    time.sleep(1)
                    print("✅ Passcode entered")
                    submit_selectors = [
                        (By.XPATH, "//button[@type='submit']"),
                        (By.XPATH, "//button[contains(text(), 'Continue')]"),
                        (By.XPATH, "//button[contains(text(), 'Submit')]"),
                    ]
                    for selector_type, selector_value in submit_selectors:
                        try:
                            submit_button = WebDriverWait(self.driver, 3).until(
                                EC.element_to_be_clickable((selector_type, selector_value))
                            )
                            submit_button.click()
                            print("✅ Submitted")
                            time.sleep(3)
                            break
                        except:
                            continue
                else:
                    print("ℹ️ No passcode field found")
            except Exception as e:
                print(f"⚠️ Passcode step: {e}")
        try:
            print("\n⏳ STEP 10: Verifying login...")
            time.sleep(3)
            success_urls = [
                "account.upstox.com",
                "developer/apps",
                "pro.upstox.com",
                "app.upstox.com"
            ]
            current_url = self.driver.current_url
            print(f"📍 Current URL: {current_url}")
            if any(url in current_url for url in success_urls):
                print("✅ LOGIN SUCCESSFUL!")
                print(f"✅ Redirected to: {current_url}")
                self.save_cookies()
                print("\n🎯 STEP 11: Navigating to Developer Apps page...")
                self.driver.get(self.target_url)
                time.sleep(3)
                print(f"✅ Navigated to: {self.target_url}")
                return True
            if "login.upstox.com" not in current_url:
                print("✅ LOGIN SUCCESSFUL!")
                self.save_cookies()
                print("\n🎯 STEP 11: Navigating to Developer Apps page...")
                self.driver.get(self.target_url)
                time.sleep(3)
                print(f"✅ Navigated to: {self.target_url}")
                return True
            print("❌ Login verification failed")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def login(self) -> bool:
        if not self.driver:
            raise RuntimeError("Driver not initialized")
        if self.load_cookies():
            self.driver.refresh()
            time.sleep(3)
            try:
                current_url = self.driver.current_url
                success_urls = ["account.upstox.com", "developer/apps", "pro.upstox.com", "app.upstox.com"]
                if any(url in current_url for url in success_urls):
                    print("♻️ Session reused")
                    if self.target_url not in current_url:
                        print(f"🎯 Navigating to: {self.target_url}")
                        self.driver.get(self.target_url)
                        time.sleep(3)
                    return True
            except:
                print("⚠️ Session expired")
        for attempt in range(1, self.max_retries + 1):
            print("\n" + "=" * 60)
            print(f"🔄 LOGIN ATTEMPT {attempt}/{self.max_retries}")
            print("=" * 60)
            if self.login_attempt():
                print("\n🎉 SUCCESS!")
                return True
            else:
                if attempt < self.max_retries:
                    print(f"\n⚠️ Attempt {attempt} failed, retrying in 5s...")
                    time.sleep(5)
        print("\n❌ All attempts failed")
        return False

    def close(self):
        if self.driver:
            self.driver.quit()
            print("🚪 Browser closed")

# ========== UPSTOX TRADER CLASS ==========
class UpstoxTrader:
    """Enhanced trader class with option chain and order management capabilities"""
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        self.order_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        # Persistent session — reuses TCP connection across all API calls
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        self._order_session = requests.Session()
        self._order_session.headers.update(self.order_headers)

    # ... (all methods unchanged from original both4) ...
    def get_user_profile(self):
        endpoint = f"{self.base_url}/user/profile"
        try:
            response = self._session.get(endpoint, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_funds(self):
        endpoint = f"{self.base_url}/user/get-funds-and-margin"
        try:
            response = self._session.get(endpoint, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_positions(self):
        endpoint = f"{self.base_url}/portfolio/short-term-positions"
        try:
            response = self._session.get(endpoint, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_order_details(self, order_id):
        endpoint = f"{self.base_url}/order/history"
        params = {"order_id": order_id}
        try:
            response = self._session.get(endpoint, params=params, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_order_book(self):
        endpoint = f"{self.base_url}/order/retrieve-all"
        try:
            response = self._session.get(endpoint, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id):
        endpoint = f"{self.base_url}/order/cancel"
        data = {"order_id": order_id}
        try:
            response = requests.delete(endpoint, headers=self.order_headers, json=data, timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def modify_order(self, order_id, trigger_price, price=0, quantity=None, order_type=None):
        """Modify an existing SL order on Upstox (used to trail the stop loss).
        
        Parameters
        ----------
        order_id      : str   – the broker order-id of the live SL order
        trigger_price : float – new SL trigger price
        price         : float – new limit price (0 for SL-M / market execution)
        quantity      : int   – lot quantity (required by Upstox modify endpoint)
        order_type    : str   – e.g. 'SL_LIMIT' or 'SL' (keep original if None)
        """
        endpoint = f"{self.base_url}/order/modify"
        payload = {
            "order_id":      order_id,
            "trigger_price": round(trigger_price, 2),
            "price":         round(price, 2),
            "validity":      "DAY",
        }
        if quantity is not None:
            payload["quantity"] = quantity
        if order_type is not None:
            payload["order_type"] = order_type.upper()
        try:
            print(f"📝 MODIFY SL ORDER: id={order_id}  new_trigger=₹{trigger_price:.2f}")
            response = self._order_session.put(endpoint, json=payload, timeout=10)
            result = response.json() if response.text else {"status": "error", "message": "Empty response"}
            if response.status_code == 200:
                print(f"✅ SL Modified → trigger ₹{trigger_price:.2f}")
            else:
                print(f"⚠️ SL Modify failed ({response.status_code}): {result}")
            return {"status_code": response.status_code, "response": result}
        except Exception as e:
            print(f"❌ SL Modify exception: {e}")
            return {"status_code": 0, "response": {"status": "error", "message": str(e)}}

    def get_ltp(self, instrument_key, max_retries=3):
        endpoint = f"{self.base_url}/market-quote/ltp"
        params = {"instrument_key": instrument_key}
        for attempt in range(max_retries):
            try:
                response = self._session.get(endpoint, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    # Upstox normalises the key in the response (e.g. NSE_EQ|xxx → NSE_EQ:xxx)
                    # Try both the original key and the colon-normalised variant
                    inner = data.get('data', {})
                    ltp_data = (inner.get(instrument_key)
                                or inner.get(instrument_key.replace('|', ':'))
                                or inner.get(instrument_key.replace(':', '|'))
                                or (list(inner.values())[0] if inner else None))
                    if ltp_data:
                        ltp = ltp_data.get('last_price')
                        if ltp and ltp > 0:
                            return ltp
                    if DEBUG_MODE and attempt == 0:
                        print(f"⚠️ LTP 200 but no price found for {instrument_key}. "
                              f"Response keys: {list(inner.keys())[:3]}")
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                if attempt < max_retries - 1 and DEBUG_MODE:
                    print(f"⚠️ LTP fetch attempt {attempt + 1} failed (status: {response.status_code}), retrying...")
                    time.sleep(2)
            except Exception as e:
                if attempt < max_retries - 1 and DEBUG_MODE:
                    print(f"⚠️ LTP fetch error (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(2)
        return None

    def get_option_chain(self, underlying_key, expiry_date=None):
        endpoint = f"{self.base_url}/option/contract"
        params = {"instrument_key": underlying_key}
        if expiry_date:
            params["expiry_date"] = expiry_date
        try:
            response = self._session.get(endpoint, params=params, timeout=15)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def place_order(self, instrument_key, quantity, transaction_type, product, order_type, price=0, trigger_price=0):
        """Place an order with Upstox API - FIXED VERSION with proper validation"""
        endpoint = f"{self.base_url}/order/place"

        # ── Service-hours guard (HTTP 423 from Upstox outside 05:30–23:59 IST) ──
        if not is_order_time_allowed():
            REJECTED_ORDER_SIGNALS.append({
                'symbol': instrument_key,
                'strategy': 'UNKNOWN',
                'reason': 'Outside Upstox service hours (05:30–23:59 IST)',
                'timestamp': datetime.now()
            })
            return {
                "status_code": 423,
                "response": {"status": "error",
                             "message": "Order blocked: outside Upstox service hours (05:30–23:59 IST)"}
            }

        # FIX 3: Validate instrument_key before placing order
        if not instrument_key or '|' not in instrument_key:
            print(f"❌ Invalid instrument_key: {instrument_key}")
            return {
                "status_code": 400,
                "response": {"status": "error", "message": f"Invalid instrument_key: {instrument_key}"}
            }
        
        payload = {
            "quantity": quantity,
            "product": product,
            "validity": "DAY",
            "price": price,
            "tag": "AUTO_BOT",
            "instrument_token": instrument_key,
            "order_type": order_type.upper(),
            "transaction_type": transaction_type.upper(),
            "disclosed_quantity": 0,
            "trigger_price": trigger_price,
            "is_amo": False
        }
        
        try:
            print(f"📤 ORDER REQUEST: {payload}")
            response = self._order_session.post(endpoint, json=payload, timeout=15)
            print(f"📥 ORDER RESPONSE ({response.status_code}): {response.text}")
            
            result = {
                "status_code": response.status_code,
                "response": response.json() if response.text else {"status": "error", "message": "Empty response"}
            }
            
            # FIX 2: Immediately show if order API failed
            if result["status_code"] != 200:
                print(f"❌ ORDER API FAILED: {result.get('response')}")
            
            return result
            
        except Exception as e:
            print(f"❌ ORDER EXCEPTION: {e}")
            return {
                "status_code": 0,
                "response": {"status": "error", "message": str(e)}
            }

# ========== HELPER FUNCTIONS ==========
def norm_key(k: str) -> str:
    """Normalize instrument keys to use pipe delimiter consistently."""
    if isinstance(k, str):
        k = k.replace(':', '|')
        if '|' in k:
            parts = k.split('|')
            if len(parts) == 2:
                return f"{parts[0]}|{parts[1]}"
    return k

def is_order_time_allowed():
    """
    Upstox API only accepts orders between 05:30 and 23:59 IST every day.
    This check is ALWAYS enforced regardless of TEST_MODE / BYPASS_MARKET_CHECKS,
    because it is an exchange-level restriction, not a market-hours restriction.
    Trying outside this window returns HTTP 423 (UDAPI100074).
    """
    now = datetime.now()
    order_start = now.replace(hour=5, minute=30, second=0, microsecond=0)
    order_end   = now.replace(hour=23, minute=59, second=59, microsecond=0)
    allowed = order_start <= now <= order_end
    if not allowed:
        print(f"⏰ Order blocked: Upstox API only accepts orders 05:30–23:59 IST "
              f"(current time {now.strftime('%H:%M:%S')})")
    return allowed

def is_market_open():
    # FIX 4: Bypass check if TEST_MODE is enabled
    if BYPASS_MARKET_CHECKS:
        return True
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    current_time = now.strftime("%H:%M")
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

def is_market_stabilized():
    # FIX 4: Bypass check if TEST_MODE is enabled
    if BYPASS_MARKET_CHECKS:
        return True
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    current_time = now.strftime("%H:%M")
    if current_time < MARKET_OPEN_TIME or current_time >= MARKET_CLOSE_TIME:
        return False
    market_open_dt = datetime.strptime(MARKET_OPEN_TIME, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
    minutes_since_open = (now - market_open_dt).total_seconds() / 60
    return minutes_since_open >= MARKET_STABILIZATION_MINUTES

def is_exit_time():
    """Check if it's time to start exiting positions"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    return current_time >= EXIT_START_TIME

def dynamic_volume_threshold():
    if not USE_DYNAMIC_VOLUME_THRESHOLD:
        return VOLUME_SPIKE_THRESHOLD
    now = datetime.now()
    market_open_dt = datetime.strptime(MARKET_OPEN_TIME, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
    minutes_since_open = (now - market_open_dt).total_seconds() / 60
    if minutes_since_open < 60:
        return 1.2
    elif minutes_since_open < 180:
        return 1.3
    else:
        return 1.5

def previous_trading_day(max_lookback_days=15):
    """Get the most recent trading day, skipping weekends and NSE holidays"""
    today = datetime.now().date()
    
    print(f"🔍 Looking for previous trading day from {today} ({today.strftime('%A')})")
    
    for d in range(1, max_lookback_days + 1):
        target_date = today - timedelta(days=d)
        
        # Skip weekends
        if target_date.weekday() >= 5:
            continue
        
        # Skip NSE holidays (check both 2025 and 2026)
        date_str = target_date.strftime('%Y-%m-%d')
        if date_str in NSE_HOLIDAYS:
            print(f"   ⚠️ Skipping {target_date} ({target_date.strftime('%A')}) - NSE Holiday")
            continue
        
        print(f"   ✅ Found: {target_date} ({target_date.strftime('%A')})")
        return target_date
    
    # Fallback
    fallback = today - timedelta(days=7)
    print(f"   ⚠️ Using fallback: {fallback}")
    return fallback

def banner():
    print("\n" + "="*120)
    print("🚀 ADVANCED TRADING SYSTEM WITH INTELLIGENT CACHING 🚀")
    print("="*120)
    print("📈 Trend-Pullback CE + S3 Breakdown + ORB Real-Time Trading System")
    print(f"Market Hours: {MARKET_OPEN_TIME}-{MARKET_CLOSE_TIME} | Stabilization: {MARKET_STABILIZATION_MINUTES}m")
    print(f"Volume: {VOLUME_SPIKE_THRESHOLD}x dynamic | Min Avg Vol: {MIN_AVG_VOLUME:,}")
    print()
    print("🎯 ACTIVE STRATEGIES:")
    print(f" • Pullback + Breakout CE: {'ENABLED ⚡' if ENABLE_AUTO_TRADING and ENABLE_PULLBACK_CE_STRATEGY else 'DISABLED'}")
    print(f" • S3 Breakdown (Buys PE): {'ENABLED ⚡' if ENABLE_AUTO_TRADING else 'DISABLED'}")
    print(f" • ORB (Opening Range Breakout): {'ENABLED ⚡' if ENABLE_ORB_STRATEGY else 'DISABLED'}")
    print()
    print(f"🔥 KLINGER OSCILLATOR:")
    print(f" • Master Filter: {'ENABLED ✓' if ENABLE_KLINGER_FILTER else 'DISABLED ✗'}")
    if ENABLE_KLINGER_FILTER:
        print(f" • Parameters: Fast={KLINGER_FAST}, Slow={KLINGER_SLOW}, Signal={KLINGER_SIGNAL}")
        print(f" • Adaptive Mode: {'ENABLED ✓' if ADAPTIVE_KLINGER_LOOKBACK else 'DISABLED'}")
        if ADAPTIVE_KLINGER_LOOKBACK:
            print(f"   - Short params: Fast={KLINGER_FAST_SHORT}, Slow={KLINGER_SLOW_SHORT}, Signal={KLINGER_SIGNAL_SHORT}")
        if KLINGER_PAPER_MODE:
            print(f" • ✗ PAPER MODE: Logging only, NOT filtering alerts")
    print()
    print(f"💾 CACHING SYSTEM:")
    if ENABLE_CANDLE_CACHE:
        print(f" • Status: ENABLED ✓")
        print(f" • Directory: {CACHE_DIRECTORY}")
        print(f" • Min Candles: {MIN_CANDLES_FOR_KLINGER}")
        print(f" • Cache Expiry: {CACHE_EXPIRY_DAYS} days")
        print(f" • Adaptive Klinger: {'ENABLED ✓' if ADAPTIVE_KLINGER_LOOKBACK else 'DISABLED'}")
    else:
        print(f" • Status: DISABLED ✗")
    print()
    print(f"🚨 EXIT MANAGEMENT:")
    print(f" • Exit Management: {'ENABLED ✓' if ENABLE_EXIT_MANAGEMENT else 'DISABLED ✗'}")
    if ENABLE_EXIT_MANAGEMENT:
        print(f" • Max Daily Loss: ₹{MAX_DAILY_LOSS:,} | Max Daily Profit: ₹{MAX_DAILY_PROFIT:,}")
        print(f" • Trailing Stop: {'ON' if ENABLE_TRAILING_STOP else 'OFF'} @ {TRAILING_STOP_PERCENTAGE}% (activates at {TRAILING_STOP_ACTIVATION}% profit)")
        print(f" • Target Multiplier: {TARGET_PROFIT_MULTIPLIER}x risk")
        print(f" • Time-based Exit: {EXIT_START_TIME} | Expiry Exit: {EXPIRY_EXIT_TIME}")
        print(f" • Position Check Interval: {POSITION_MONITORING_INTERVAL}s")
    print()
    print(f"⚙️ SETTINGS:")
    print(f" Max Orders/Day: {MAX_ORDERS_PER_DAY} | Size: {ORDER_QUANTITY} LOTS")
    print(f" Product: {ORDER_PRODUCT} | Premium Safety SL: {STOPLOSS_PERCENTAGE}% (Pullback CE fallback: {PULLBACK_OPTION_SAFETY_SL_PERCENT}%)")
    print("="*120 + "\n")

def verify_token(token, verbose=True):
    """Verify API token and return validation status"""
    if verbose:
        print("🔍 Verifying API token...")

    # --- PRE-CHECK: Decode JWT expiry without making API call ---
    try:
        import base64, json as _json
        parts = token.split('.')
        if len(parts) == 3:
            payload_b64 = parts[1] + '=' * (4 - len(parts[1]) % 4)
            payload = _json.loads(base64.b64decode(payload_b64).decode('utf-8'))
            exp_ts = payload.get('exp')
            if exp_ts:
                exp_dt = datetime.fromtimestamp(exp_ts)
                now = datetime.now()
                if now > exp_dt:
                    print(f"🚨 TOKEN EXPIRED at {exp_dt.strftime('%Y-%m-%d %H:%M:%S')} — IT IS NOW {now.strftime('%H:%M:%S')} — ORDERS WILL FAIL!")
                    print(f"   ➡ Set USE_HARDCODED_TOKEN=False to auto-login, or update HARDCODED_TOKEN")
                else:
                    mins_left = int((exp_dt - now).total_seconds() / 60)
                    if verbose:
                        print(f"⏱ Token expires at {exp_dt.strftime('%H:%M:%S')} ({mins_left} min remaining)")
    except Exception:
        pass  # Don't block on JWT decode failure
    # --- END PRE-CHECK ---

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    url = "https://api.upstox.com/v2/user/profile"
    try:
        response = _get_upstox_session(token).get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if verbose:
                print("✅ Token is VALID")
                if 'data' in data:
                    user_name = data['data'].get('user_name', 'N/A')
                    user_id = data['data'].get('user_id', 'N/A')
                    print(f" User: {user_name} (ID: {user_id})\n")
            return {
                'valid': True,
                'data': data.get('data', {}),
                'message': 'Token is valid'
            }
        elif response.status_code == 401:
            if verbose:
                print("❌ Token is INVALID or EXPIRED")
            return {
                'valid': False,
                'message': 'Token is invalid or expired',
                'status_code': 401
            }
        else:
            if verbose:
                print(f"⚠️ Unexpected response: {response.status_code}")
            return {
                'valid': False,
                'message': f'Unexpected status code: {response.status_code}',
                'status_code': response.status_code
            }
    except requests.exceptions.Timeout:
        if verbose:
            print("❌ Token verification timed out")
        return {
            'valid': False,
            'message': 'Request timeout'
        }
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"❌ Token verification failed: {e}")
        return {
            'valid': False,
            'message': f'Request failed: {str(e)}'
        }
    except Exception as e:
        if verbose:
            print(f"❌ Token verification error: {e}")
        return {
            'valid': False,
            'message': f'Verification error: {str(e)}'
        }

# ============================================================================
# FII/DII EXTRACTION AND ORB STRATEGY FUNCTIONS
# ============================================================================
# (These functions remain exactly as in original both4; they are not modified by the cache)
def extract_fii_dii_data():
    global FII_DII_DATA, FII_DII_LAST_UPDATE, FII_DII_STRONG_BUY, FII_DII_STRONG_SELL, FII_DII_MIXED
    print(f"\n{'='*100}")
    print("🔍 EXTRACTING FII/DII DATA FROM MUNAFASUTRA")
    print(f"{'='*100}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(FII_DII_URL, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        container = soup.find('div', {'id': 'allFIIDII'})
        if not container:
            container = soup.find('div', {'class': 'wideTable'})
        table = container.find('table') if container else soup.find('table')
        if not table:
            print("❌ Table not found")
            return load_fii_dii_from_cache()
        stocks = []
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                company_cell = cols[0]
                link = company_cell.find('a')
                text = link.get_text(strip=True) if link else company_cell.get_text(strip=True)
                if '(' in text and ')' in text:
                    symbol = text.split('(')[-1].replace(')', '').strip()
                    name = text.split('(')[0].strip()
                else:
                    name = text
                    symbol = ""
                if not symbol:
                    continue
                stock = {
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Symbol': symbol,
                    'Stock_Name': name,
                    'FII_DII_Cash': cols[1].get_text(strip=True),
                    'FII_DII_FNO': cols[2].get_text(strip=True),
                    'Price_Change': cols[3].get_text(strip=True),
                    'Current_Price': cols[4].get_text(strip=True).replace(',', '')
                }
                stocks.append(stock)
        if not stocks:
            return load_fii_dii_from_cache()
        df = pd.DataFrame(stocks)
        filename = f"FII_DII_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        FII_DII_DATA = {row['Symbol']: row for _, row in df.iterrows()}
        FII_DII_LAST_UPDATE = datetime.now()
        FII_DII_STRONG_BUY = set(df[(df['FII_DII_Cash'] == 'Bought') & (df['FII_DII_FNO'] == 'Bought')]['Symbol'].values)
        FII_DII_STRONG_SELL = set(df[(df['FII_DII_Cash'] == 'Sold') & (df['FII_DII_FNO'] == 'Sold')]['Symbol'].values)
        FII_DII_MIXED = set(df['Symbol'].values) - FII_DII_STRONG_BUY - FII_DII_STRONG_SELL
        print(f"✅ Extracted {len(df)} stocks")
        print(f"💪 STRONG BUY: {len(FII_DII_STRONG_BUY)} stocks")
        print(f"🔴 STRONG SELL: {len(FII_DII_STRONG_SELL)} stocks")
        print(f"⚠️  MIXED: {len(FII_DII_MIXED)} stocks")
        if FII_DII_STRONG_BUY:
            print(f"\n📈 Top 10 Strong Buy Stocks:")
            strong_buy_df = df[df['Symbol'].isin(FII_DII_STRONG_BUY)].copy()
            strong_buy_df['PC'] = strong_buy_df['Price_Change'].str.replace('%', '').astype(float)
            for _, row in strong_buy_df.nlargest(10, 'PC').iterrows():
                print(f"   {row['Symbol']:12} | {row['Price_Change']:>7} | ₹{row['Current_Price']}")
        save_fii_dii_to_cache()
        # Run multi-day trend analysis after saving today's data
        if ENABLE_FII_DII_TREND_FILTER:
            analyze_fii_dii_trends()
        return df
    except Exception as e:
        print(f"❌ Error extracting FII/DII: {e}")
        return load_fii_dii_from_cache()

# FIXED: save_fii_dii_to_cache now handles pandas Series correctly
def save_fii_dii_to_cache():
    try:
        # Convert any Series/DataFrame values to plain dicts
        serializable_data = {}
        for k, v in FII_DII_DATA.items():
            if hasattr(v, 'to_dict'):
                serializable_data[k] = v.to_dict()
            elif isinstance(v, dict):
                serializable_data[k] = {
                    dk: (dv.item() if hasattr(dv, 'item') else dv) 
                    for dk, dv in v.items()
                }
            else:
                serializable_data[k] = v

        with open(FII_DII_CACHE_FILE, 'w') as f:
            json.dump({
                'data': serializable_data,
                'strong_buy': list(FII_DII_STRONG_BUY),
                'strong_sell': list(FII_DII_STRONG_SELL),
                'mixed': list(FII_DII_MIXED),
                'last_update': FII_DII_LAST_UPDATE.isoformat() if FII_DII_LAST_UPDATE else None
            }, f)
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️ Cache save error: {e}")

def load_fii_dii_from_cache():
    global FII_DII_DATA, FII_DII_LAST_UPDATE, FII_DII_STRONG_BUY, FII_DII_STRONG_SELL, FII_DII_MIXED
    if os.path.exists(FII_DII_CACHE_FILE):
        try:
            print("📂 Loading FII/DII from cache...")
            with open(FII_DII_CACHE_FILE, 'r') as f:
                cache = json.load(f)
                FII_DII_DATA = cache.get('data', {})
                FII_DII_STRONG_BUY = set(cache.get('strong_buy', []))
                FII_DII_STRONG_SELL = set(cache.get('strong_sell', []))
                FII_DII_MIXED = set(cache.get('mixed', []))
                if cache.get('last_update'):
                    FII_DII_LAST_UPDATE = datetime.fromisoformat(cache['last_update'])
                print(f"✅ Loaded cache from {FII_DII_LAST_UPDATE}")
                # Also restore trend sets from their own cache file
                if ENABLE_FII_DII_TREND_FILTER:
                    _load_fii_dii_trend_cache()
                return pd.DataFrame(list(FII_DII_DATA.values())) if FII_DII_DATA else None
        except Exception as e:
            if DEBUG_MODE:
                print(f"❌ Cache load error: {e}")
    return None


def analyze_fii_dii_trends():
    """
    Load all historical FII_DII_YYYYMMDD.csv files and detect multi-day
    institutional patterns. Updates the four global trend sets.

    Patterns detected:
      STRONG_ACCUMULATION : Both FII cash + FNO bought today
      FII_BUY_DII_SELL    : FII cash bought, FNO sold (FII leading momentum)
      FII_SELL_DII_BUY    : FII cash sold, FNO bought (DII absorbing selling)
      UNUSUAL_CHANGE      : Today reversed vs previous day on both legs

    Also persists results to FII_DII_TREND_CACHE_FILE so they survive restarts.
    """
    global FII_DII_TREND_STRONG_ACCUMULATION, FII_DII_TREND_FII_BUY_DII_SELL
    global FII_DII_TREND_FII_SELL_DII_BUY, FII_DII_TREND_UNUSUAL_CHANGE

    import glob

    try:
        files = sorted(glob.glob("FII_DII_*.csv"))
        if len(files) < 1:
            if DEBUG_MODE:
                print("FII/DII trend: no CSV files found -- trends unavailable")
            _load_fii_dii_trend_cache()
            return

        df_list = []
        for fp in files:
            date_str = fp.replace('FII_DII_', '').replace('.csv', '')
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d').date()
            except Exception:
                continue
            try:
                tmp = pd.read_csv(fp)
                tmp['Date'] = file_date
                df_list.append(tmp)
            except Exception:
                continue

        if not df_list:
            _load_fii_dii_trend_cache()
            return

        combined = pd.concat(df_list, ignore_index=True)

        # Fix 2: Drop duplicate (Date, Symbol) rows that appear when the bot is
        # run multiple times on the same day — keep the last occurrence (most recent).
        if 'Symbol' in combined.columns and 'Date' in combined.columns:
            combined = combined.drop_duplicates(subset=['Date', 'Symbol'], keep='last')

        # Fix 4: Price_Change conversion is not used downstream — only convert
        # if the column exists, inside a guard so missing column never raises.
        # (Conversion kept for completeness; result not currently referenced.)
        if 'Price_Change' in combined.columns:
            combined['Price_Change'] = pd.to_numeric(
                combined['Price_Change'].astype(str)
                    .str.replace('%', '', regex=False)
                    .str.strip(),
                errors='coerce'
            )

        dates = sorted(combined['Date'].unique())
        latest_date = dates[-1]
        prev_date   = dates[-2] if len(dates) >= 2 else None

        latest = combined[combined['Date'] == latest_date]
        prev   = combined[combined['Date'] == prev_date] if prev_date else pd.DataFrame()

        strong_acc   = set()
        fii_buy_sell = set()
        fii_sell_buy = set()
        unusual      = set()

        for _, row in latest.iterrows():
            symbol = row.get('Symbol', '')
            if not symbol:
                continue
            cash = str(row.get('FII_DII_Cash', '')).strip()
            fno  = str(row.get('FII_DII_FNO',  '')).strip()

            if cash == 'Bought' and fno == 'Bought':
                strong_acc.add(symbol)
            elif cash == 'Bought' and fno == 'Sold':
                fii_buy_sell.add(symbol)
            elif cash == 'Sold' and fno == 'Bought':
                fii_sell_buy.add(symbol)

            if not prev.empty:
                prev_row = prev[prev['Symbol'] == symbol]
                if not prev_row.empty:
                    prev_cash = str(prev_row.iloc[0].get('FII_DII_Cash', '')).strip()
                    prev_fno  = str(prev_row.iloc[0].get('FII_DII_FNO',  '')).strip()
                    # Pattern 1: Sudden institutional buy (both sold → both bought)
                    if (prev_cash, prev_fno) == ('Sold', 'Sold') and (cash, fno) == ('Bought', 'Bought'):
                        unusual.add(symbol)
                    # Pattern 2: FII flip to lead (FII sell/DII buy → FII buy/DII sell)
                    if (prev_cash, prev_fno) == ('Sold', 'Bought') and (cash, fno) == ('Bought', 'Sold'):
                        unusual.add(symbol)
                    # Pattern 3: Sudden institutional exit (both bought → both sold)
                    if (prev_cash, prev_fno) == ('Bought', 'Bought') and (cash, fno) == ('Sold', 'Sold'):
                        unusual.add(symbol)
                    # Pattern 4: DII flip to lead (FII buy/DII sell → FII sell/DII buy)
                    if (prev_cash, prev_fno) == ('Bought', 'Sold') and (cash, fno) == ('Sold', 'Bought'):
                        unusual.add(symbol)

        with FII_DII_TREND_LOCK:
            FII_DII_TREND_STRONG_ACCUMULATION = strong_acc
            FII_DII_TREND_FII_BUY_DII_SELL    = fii_buy_sell
            FII_DII_TREND_FII_SELL_DII_BUY    = fii_sell_buy
            FII_DII_TREND_UNUSUAL_CHANGE      = unusual

        _save_fii_dii_trend_cache()

        print(f"\n FII/DII TREND ANALYSIS (using {len(files)} days of data):")
        print(f"   Strong Accumulation : {len(strong_acc)} stocks"
              + (f" -- {', '.join(sorted(strong_acc)[:8])}" if strong_acc else ""))
        print(f"   FII Buy / DII Sell  : {len(fii_buy_sell)} stocks"
              + (f" -- {', '.join(sorted(fii_buy_sell)[:8])}" if fii_buy_sell else ""))
        print(f"   FII Sell / DII Buy  : {len(fii_sell_buy)} stocks"
              + (f" -- {', '.join(sorted(fii_sell_buy)[:8])}" if fii_sell_buy else ""))
        print(f"   Unusual Reversal    : {len(unusual)} stocks"
              + (f" -- {', '.join(sorted(unusual)[:8])}" if unusual else ""))

    except Exception as e:
        print(f"FII/DII trend analysis error: {e}")
        _load_fii_dii_trend_cache()


def _save_fii_dii_trend_cache():
    """Persist trend sets to JSON so they survive restarts."""
    try:
        with FII_DII_TREND_LOCK:
            payload = {
                'strong_accumulation': list(FII_DII_TREND_STRONG_ACCUMULATION),
                'fii_buy_dii_sell':    list(FII_DII_TREND_FII_BUY_DII_SELL),
                'fii_sell_dii_buy':    list(FII_DII_TREND_FII_SELL_DII_BUY),
                'unusual_change':      list(FII_DII_TREND_UNUSUAL_CHANGE),
                'saved_at':            datetime.now().isoformat(),
            }
        with open(FII_DII_TREND_CACHE_FILE, 'w') as f:
            json.dump(payload, f)
    except Exception as e:
        if DEBUG_MODE:
            print(f"FII/DII trend cache save error: {e}")


def _load_fii_dii_trend_cache():
    """Load trend sets from JSON cache (fallback when CSV files unavailable)."""
    global FII_DII_TREND_STRONG_ACCUMULATION, FII_DII_TREND_FII_BUY_DII_SELL
    global FII_DII_TREND_FII_SELL_DII_BUY, FII_DII_TREND_UNUSUAL_CHANGE
    if not os.path.exists(FII_DII_TREND_CACHE_FILE):
        return
    try:
        with open(FII_DII_TREND_CACHE_FILE) as f:
            c = json.load(f)
        with FII_DII_TREND_LOCK:
            FII_DII_TREND_STRONG_ACCUMULATION = set(c.get('strong_accumulation', []))
            FII_DII_TREND_FII_BUY_DII_SELL    = set(c.get('fii_buy_dii_sell',    []))
            FII_DII_TREND_FII_SELL_DII_BUY    = set(c.get('fii_sell_dii_buy',    []))
            FII_DII_TREND_UNUSUAL_CHANGE      = set(c.get('unusual_change',      []))
        print(f"FII/DII trend cache loaded (saved: {c.get('saved_at','?')})")
    except Exception as e:
        if DEBUG_MODE:
            print(f"FII/DII trend cache load error: {e}")


def get_fii_dii_trend_score(symbol):
    """
    Return an integer score reflecting multi-day FII/DII institutional stance.
    Scores stack -- e.g. strong accumulation + unusual change = +4.

      +2 : Strong accumulation (both bought today)
      +2 : Unusual reversal (sudden buy after sell)
      +1 : FII leading buy (cash bought, FNO sold)
      -1 : FII distributing (cash sold, FNO bought)
       0 : No trend data / neutral
    """
    score = 0
    with FII_DII_TREND_LOCK:
        if symbol in FII_DII_TREND_STRONG_ACCUMULATION:
            score += FII_DII_SCORE_STRONG_ACC
        if symbol in FII_DII_TREND_FII_BUY_DII_SELL:
            score += FII_DII_SCORE_FII_BUY
        if symbol in FII_DII_TREND_FII_SELL_DII_BUY:
            score += FII_DII_SCORE_FII_SELL
        if symbol in FII_DII_TREND_UNUSUAL_CHANGE:
            score += FII_DII_SCORE_UNUSUAL
    return score


def get_fii_dii_signal(symbol):
    if symbol in FII_DII_STRONG_BUY:
        return 'STRONG_BUY'
    elif symbol in FII_DII_STRONG_SELL:
        return 'STRONG_SELL'
    elif symbol in FII_DII_DATA:
        data = FII_DII_DATA[symbol]
        if data['FII_DII_Cash'] == 'Bought' or data['FII_DII_FNO'] == 'Bought':
            return 'BUY'
        elif data['FII_DII_Cash'] == 'Sold' or data['FII_DII_FNO'] == 'Sold':
            return 'SELL'
    return 'NEUTRAL'

def calculate_orb_levels(symbol, open_price, close_price, high_price, low_price, volume,
                         candle_df=None, instrument_key=None):
    """
    Calculate ORB levels with Klinger + RSI quality gate.

    Args:
        candle_df:      optional DataFrame of recent 5-min candles (for RSI gate).
                        If None, the function tries get_realtime_5min_df(symbol).
        instrument_key: NSE_EQ instrument_key used to look up Klinger from R3_LEVELS.
                        If None, falls back to SYMBOL_TO_ISIN.get(symbol).
    """
    body_size = abs(close_price - open_price)
    body_percent = (body_size / open_price) * 100
    is_bullish = close_price > open_price
    is_bearish = close_price < open_price
    if not is_bullish and not is_bearish:
        return None

    # Directional body threshold (slightly higher to filter weak candles)
    min_body = ORB_MIN_CANDLE_BODY_LONG if is_bullish else ORB_MIN_CANDLE_BODY_SHORT
    if body_percent < min_body:
        return None

    if is_bullish:
        breakout_level = close_price
        stop_level     = low_price   # Use candle LOW (tighter, more accurate than open)
        target_level   = close_price + (body_size * ORB_TARGET_MULTIPLIER)
        direction      = 'BUY'
        signal_type    = 'BULLISH_ORB'
    else:
        breakout_level = close_price
        stop_level     = high_price  # Use candle HIGH for shorts
        target_level   = close_price - (body_size * ORB_TARGET_MULTIPLIER)
        direction      = 'SELL'
        signal_type    = 'BEARISH_ORB'

    fii_dii_signal = get_fii_dii_signal(symbol)

    # ── FII/DII confidence ──────────────────────────────────────────────────
    if is_bullish and fii_dii_signal == 'STRONG_BUY':
        confidence = 'VERY_HIGH'
    elif not is_bullish and fii_dii_signal == 'STRONG_SELL':
        confidence = 'VERY_HIGH'
    elif is_bullish and fii_dii_signal == 'BUY':
        confidence = 'HIGH'
    elif not is_bullish and fii_dii_signal == 'SELL':
        confidence = 'HIGH'
    else:
        confidence = 'MEDIUM'

    if ORB_ENABLE_FII_DII_FILTER and confidence == 'MEDIUM':
        return None

    # ── Resolve instrument_key for Klinger lookup ────────────────────────────
    # R3_LEVELS is keyed by instrument_key (e.g. NSE_EQ|INE...), not symbol.
    ikey = instrument_key or SYMBOL_TO_ISIN.get(symbol)

    # ── Klinger gate ────────────────────────────────────────────────────────
    if ORB_ENABLE_KLINGER_GATE:
        klinger_info = R3_LEVELS.get(ikey, {}).get('klinger') if ikey else None
        ko = klinger_info.get('klinger') if klinger_info else None
        if ko is not None:
            if is_bullish and ko < 0:
                # Klinger still negative — only allow VERY_HIGH (strong FII alignment)
                if confidence != 'VERY_HIGH':
                    if DEBUG_MODE:
                        print(f"⛔ ORB KLINGER gate: {symbol} LONG suppressed "
                              f"(KO={ko:,.0f} < 0, confidence={confidence})")
                    return None
            elif not is_bullish and ko > 0:
                if confidence != 'VERY_HIGH':
                    if DEBUG_MODE:
                        print(f"⛔ ORB KLINGER gate: {symbol} SHORT suppressed "
                              f"(KO={ko:,.0f} > 0, confidence={confidence})")
                    return None

    # ── RSI gate ─────────────────────────────────────────────────────────────
    # Use caller-supplied candle_df, or fall back to the real-time builder.
    rsi_value = None
    if ORB_ENABLE_RSI_GATE:
        df_for_rsi = candle_df
        if df_for_rsi is None:
            df_for_rsi = get_realtime_5min_df(symbol, min_bars=15)
        if df_for_rsi is not None and len(df_for_rsi) >= 15:
            try:
                rsi_value = calculate_rsi(df_for_rsi, period=14)
            except Exception:
                rsi_value = None

    if ORB_ENABLE_RSI_GATE and rsi_value is not None:
        if is_bullish and rsi_value < ORB_RSI_LONG_MIN:
            if confidence != 'VERY_HIGH':
                if DEBUG_MODE:
                    print(f"⛔ ORB RSI gate: {symbol} LONG suppressed "
                          f"(RSI={rsi_value:.1f} < {ORB_RSI_LONG_MIN}, confidence={confidence})")
                return None
        elif not is_bullish and rsi_value > ORB_RSI_SHORT_MAX:
            if confidence != 'VERY_HIGH':
                if DEBUG_MODE:
                    print(f"⛔ ORB RSI gate: {symbol} SHORT suppressed "
                          f"(RSI={rsi_value:.1f} > {ORB_RSI_SHORT_MAX}, confidence={confidence})")
                return None

    risk   = abs(breakout_level - stop_level)
    reward = abs(target_level   - breakout_level)
    if risk <= 0:
        return None

    # Snapshot Klinger KO value for logging (already resolved above)
    klinger_info_snap = R3_LEVELS.get(ikey, {}).get('klinger') if ikey else None
    ko_snap = klinger_info_snap.get('klinger') if klinger_info_snap else None

    result = {
        'symbol':          symbol,
        'instrument_key':  ikey,
        'timestamp':       datetime.now(),
        'signal_type':     signal_type,
        'direction':       direction,
        'open':            open_price,
        'close':           close_price,
        'high':            high_price,
        'low':             low_price,
        'body_size':       body_size,
        'body_percent':    body_percent,
        'breakout_level':  breakout_level,
        'stop_level':      stop_level,
        'target_level':    target_level,
        'volume':          volume,
        'is_bullish':      is_bullish,
        'risk':            risk,
        'reward':          reward,
        'risk_reward':     reward / risk,
        'fii_dii_signal':  fii_dii_signal,
        'confidence':      confidence,
        'rsi_at_signal':   rsi_value,
        'klinger_at_signal': ko_snap,
    }
    return result

def process_first_candles(access_token, live_data, late_pass=False):
    """
    Build ORB signals from the first 5-minute candle.

    Called in two modes:
      1. Primary pass (late_pass=False) at 09:20–09:25 — processes all stocks.
         Symbols with zero volume are added to ORB_LATE_CHECKED for retry.
      2. Late pass (late_pass=True) at 09:25 until the breakout window closes —
         only retries symbols in ORB_LATE_CHECKED that still lack a signal and
         now have volume.  Once a symbol gets a signal (or volume stays zero all
         the way to the window close) it is removed from ORB_LATE_CHECKED.
    """
    global ORB_SIGNALS, ORB_PROCESSED_TODAY, ORB_LATE_CHECKED
    if not late_pass:
        print(f"\n{'='*100}")
        print("📊 PROCESSING FIRST 5-MINUTE CANDLES FOR ORB STRATEGY")
        print(f"{'='*100}\n")
        ORB_LATE_CHECKED.clear()          # fresh slate each trading day
    else:
        if not ORB_LATE_CHECKED:
            return                        # nothing left to retry — skip immediately
        print(f"\n🔄 ORB late-volume pass ({datetime.now().strftime('%H:%M')}) — "
              f"retrying {len(ORB_LATE_CHECKED)} zero-volume symbols from 09:20")

    orb_count = 0
    very_high = 0
    high = 0

    # Late pass iterates only the small ORB_LATE_CHECKED set, not all 171 stocks
    candidates = (
        {sk: live_data[sk] for sk in list(ORB_LATE_CHECKED) if sk in live_data}
        if late_pass else live_data
    )

    for symbol_key, data in candidates.items():
        try:
            symbol = ISIN_TO_SYMBOL.get(symbol_key, symbol_key)
            ltp    = data.get('ltp', 0)
            volume = data.get('volume', 0)

            if volume == 0:
                if not late_pass:
                    ORB_LATE_CHECKED.add(symbol_key)   # remember for retry
                continue

            # Symbol now has volume — remove from retry set regardless of outcome
            ORB_LATE_CHECKED.discard(symbol_key)

            open_price  = data.get('open', ltp)
            close_price = ltp
            high_price  = data.get('high', ltp)
            low_price   = data.get('low', ltp)

            candle_df  = get_realtime_5min_df(symbol, min_bars=15)
            orb_signal = calculate_orb_levels(
                symbol, open_price, close_price, high_price, low_price, volume,
                candle_df=candle_df, instrument_key=symbol_key
            )
            if orb_signal:
                ORB_SIGNALS[symbol] = orb_signal
                orb_count += 1
                if orb_signal['confidence'] == 'VERY_HIGH':
                    very_high += 1
                elif orb_signal['confidence'] == 'HIGH':
                    high += 1
                log_orb_signal(orb_signal)
                if orb_signal['confidence'] in ['VERY_HIGH', 'HIGH']:
                    rsi_str = f"RSI={orb_signal['rsi_at_signal']:.1f}" if orb_signal['rsi_at_signal'] else "RSI=N/A"
                    ko      = orb_signal['klinger_at_signal']
                    ko_str  = f"KO={ko/1e6:.1f}M" if ko is not None else "KO=N/A"
                    tag     = " [LATE]" if late_pass else ""
                    print(f"✅{tag} {symbol:12} | {orb_signal['signal_type']:15} | "
                          f"{orb_signal['confidence']:10} | FII: {orb_signal['fii_dii_signal']:12} | "
                          f"R:R {orb_signal['risk_reward']:.2f}:1 | {rsi_str} | {ko_str}")
        except Exception as e:
            if DEBUG_MODE:
                print(f"ORB Error {symbol_key}: {e}")

    ORB_PROCESSED_TODAY = True
    if orb_count > 0 or not late_pass:
        print(f"\n✅ {'Late pass:' if late_pass else 'Processed'} {orb_count} ORB signals"
              f"{'' if late_pass else f' | VERY HIGH: {very_high} | HIGH: {high}'}"
              f"{f' | Still awaiting volume: {len(ORB_LATE_CHECKED)}' if late_pass and ORB_LATE_CHECKED else ''}")
    if not late_pass:
        if ORB_LATE_CHECKED:
            print(f"   ⏳ {len(ORB_LATE_CHECKED)} stocks had zero volume — will retry until breakout window closes")
        print(f"{'='*100}\n")

def check_orb_breakout(symbol, current_price, current_volume, live_data):
    if symbol not in ORB_SIGNALS or symbol in ORB_ALERTED_STOCKS:
        return None
    orb = ORB_SIGNALS[symbol]
    now = datetime.now()
    market_open_920 = now.replace(hour=9, minute=20, second=0, microsecond=0)
    minutes_since_920 = (now - market_open_920).total_seconds() / 60
    if minutes_since_920 < 0 or minutes_since_920 > ORB_BREAKOUT_WINDOW_MINUTES:
        return None

    # ── Volume check: use VOLUME_DATA OR live_data avg_volume ───────────────
    avg_volume = (VOLUME_DATA.get(symbol, {}).get('avg_volume')
                  or VOLUME_DATA.get(symbol, {}).get('avg_vol_20d')
                  or live_data.get('avg_volume', 0))
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    if avg_volume > 0 and volume_ratio < ORB_VOLUME_CONFIRMATION:
        return None

    # ── Live RSI re-check at breakout moment ────────────────────────────────
    if ORB_ENABLE_RSI_GATE:
        try:
            df = get_realtime_5min_df(symbol, min_bars=15)
            if df is not None and len(df) >= 15:
                live_rsi = calculate_rsi(df, period=14)
                if live_rsi is not None:
                    if orb['is_bullish'] and live_rsi < ORB_RSI_LONG_MIN:
                        if orb['confidence'] != 'VERY_HIGH':
                            if DEBUG_MODE:
                                print(f"⛔ ORB breakout RSI gate: {symbol} LONG "
                                      f"(RSI={live_rsi:.1f} < {ORB_RSI_LONG_MIN} at breakout)")
                            return None
                    elif not orb['is_bullish'] and live_rsi > ORB_RSI_SHORT_MAX:
                        if orb['confidence'] != 'VERY_HIGH':
                            if DEBUG_MODE:
                                print(f"⛔ ORB breakout RSI gate: {symbol} SHORT "
                                      f"(RSI={live_rsi:.1f} > {ORB_RSI_SHORT_MAX} at breakout)")
                            return None
        except Exception:
            pass

    breakout_signal = None
    if orb['is_bullish'] and current_price > orb['breakout_level'] * 1.001:
        breakout_signal = {
            'symbol':        symbol,
            'signal':        'ORB_BREAKOUT',
            'direction':     'BUY',
            'entry_price':   current_price,
            'stop_loss':     orb['stop_level'],
            'target':        orb['target_level'],
            'orb_data':      orb,
            'volume_ratio':  volume_ratio,
            'confidence':    orb['confidence'],
            'fii_dii_signal':orb['fii_dii_signal'],
            'risk':          orb['risk'],
            'reward':        orb['reward'],
            'risk_reward':   orb['risk_reward'],
            'entry_type':    ENTRY_ORB_BULLISH
        }
    elif not orb['is_bullish'] and current_price < orb['breakout_level'] * 0.999:
        breakout_signal = {
            'symbol':        symbol,
            'signal':        'ORB_BREAKDOWN',
            'direction':     'SELL',
            'entry_price':   current_price,
            'stop_loss':     orb['stop_level'],
            'target':        orb['target_level'],
            'orb_data':      orb,
            'volume_ratio':  volume_ratio,
            'confidence':    orb['confidence'],
            'fii_dii_signal':orb['fii_dii_signal'],
            'risk':          orb['risk'],
            'reward':        orb['reward'],
            'risk_reward':   orb['risk_reward'],
            'entry_type':    ENTRY_ORB_BEARISH
        }
    return breakout_signal

def log_orb_signal(signal):
    try:
        with open(ORB_SIGNALS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signal['symbol'],
                signal['signal_type'],
                signal['direction'],
                f"{signal['breakout_level']:.2f}",
                f"{signal['stop_level']:.2f}",
                f"{signal['target_level']:.2f}",
                f"{signal['body_percent']:.2f}",
                f"{signal['risk_reward']:.2f}",
                signal['fii_dii_signal'],
                signal['confidence']
            ])
    except:
        pass

def log_orb_trade(trade, action='ENTRY'):
    try:
        with open(ORB_TRADES_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                trade['symbol'],
                action,
                trade['direction'],
                f"{trade['entry_price']:.2f}",
                f"{trade['stop_loss']:.2f}",
                f"{trade['target']:.2f}",
                f"{trade.get('volume_ratio', 0):.2f}",
                trade['confidence'],
                trade['fii_dii_signal']
            ])
    except:
        pass

def send_orb_alert(signal, trader=None):
    global ORB_ALERTED_STOCKS, ORB_ORDER_COUNT
    ORB_ALERTED_STOCKS.add(signal['symbol'])
    print("\n" + "="*100)
    print(f"⚡ ORB SIGNAL: {signal['symbol']} ⚡")
    print("="*100)
    print(f"Signal:       {signal['signal']}")
    print(f"Direction:    {signal['direction']}")
    print(f"Confidence:   {signal['confidence']}")
    print(f"FII/DII:      {signal['fii_dii_signal']}")
    print(f"Entry Price:  ₹{signal['entry_price']:.2f}")
    print(f"Stop Loss:    ₹{signal['stop_loss']:.2f}")
    print(f"Target:       ₹{signal['target']:.2f}")
    print(f"Risk:         ₹{signal['risk']:.2f} per share")
    print(f"Reward:       ₹{signal['reward']:.2f} per share")
    print(f"R:R Ratio:    {signal['risk_reward']:.2f}:1")
    print(f"Volume:       {signal['volume_ratio']:.2f}x average")
    orb_d = signal.get('orb_data', {})
    rsi_val = orb_d.get('rsi_at_signal')
    ko_val  = orb_d.get('klinger_at_signal')
    if rsi_val is not None:
        print(f"RSI at entry: {rsi_val:.1f}")
    if ko_val is not None:
        print(f"Klinger (KO): {ko_val:,.0f}")
    print("="*100)
    log_orb_trade(signal, 'ENTRY')
    with open(ORB_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"ORB ALERT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: {signal['symbol']}\n")
        f.write(f"Signal: {signal['signal']} ({signal['direction']})\n")
        f.write(f"Confidence: {signal['confidence']} | FII/DII: {signal['fii_dii_signal']}\n")
        f.write(f"Entry: ₹{signal['entry_price']:.2f} | Stop: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['target']:.2f}\n")
        f.write(f"R:R: {signal['risk_reward']:.2f}:1 | Volume: {signal['volume_ratio']:.2f}x\n")
        f.write(f"{'='*100}\n")
    try:
        with open(ALERT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signal['symbol'],
                f"ORB_{signal['direction']}",
                signal['entry_price'],
                signal['volume_ratio'],
                '',
                signal['confidence'],
                signal['fii_dii_signal']
            ])
    except:
        pass
    if ENABLE_AUTO_TRADING and trader and ORB_ORDER_COUNT < MAX_ORDERS_PER_DAY:
        if not is_order_time_allowed():
            print(f"⏭️  ORB {signal['symbol']}: order skipped — outside Upstox service hours (05:30–23:59 IST)")
            return
        print(f"\n📤 Placing ORB {signal['direction']} order for {signal['symbol']}...")
        orb_breakout = {
            'symbol':        signal['symbol'],
            'instrument_key': signal.get('instrument_key', signal.get('orb_data', {}).get('instrument_key', '')),
            'breakout_type': 'CE' if signal['direction'] == 'BUY' else 'PE',
            'strategy':      'ORB',
            'entry_price':   signal['entry_price'],
            'stop_loss':     signal['stop_loss'],
            'target':        signal['target'],
            'klinger_status': signal.get('orb_data', {}).get('klinger_at_signal'),
        }
        order_id = place_breakout_order(orb_breakout, trader)
        if order_id:
            ORB_ORDER_COUNT += 1
            print(f"✅ ORB order placed: {order_id} | ORB orders today: {ORB_ORDER_COUNT}/{MAX_ORDERS_PER_DAY}")
        else:
            print(f"⚠️ ORB order failed for {signal['symbol']}")
    elif ORB_ORDER_COUNT >= MAX_ORDERS_PER_DAY:
        print(f"⚠️ ORB order limit reached ({ORB_ORDER_COUNT}/{MAX_ORDERS_PER_DAY}) — signal logged only")

def initialize_orb_csv_files():
    csv_files = [
        (ORB_SIGNALS_FILE, ['Timestamp', 'Symbol', 'Signal_Type', 'Direction', 
                           'Breakout_Level', 'Stop_Level', 'Target_Level', 
                           'Body_Percent', 'Risk_Reward', 'FII_DII_Signal', 'Confidence']),
        (ORB_TRADES_FILE, ['Timestamp', 'Symbol', 'Action', 'Direction', 'Price', 
                          'Stop_Loss', 'Target', 'Volume_Ratio', 'Confidence', 'FII_DII_Signal'])
    ]
    for csv_file, headers in csv_files:
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

def update_fii_dii_if_needed():
    global FII_DII_LAST_UPDATE
    if not ENABLE_FII_DII_FILTER:
        return
    if FII_DII_LAST_UPDATE is None:
        extract_fii_dii_data()
        return
    # FII/DII data is published once per day after market close.
    # Only re-fetch if the existing data is from a previous calendar day,
    # or if more than FII_DII_UPDATE_INTERVAL seconds have passed (safety net).
    last_date = FII_DII_LAST_UPDATE.date()
    today     = datetime.now().date()
    if last_date < today:
        print(f"\n🔄 FII/DII data is from {last_date} — fetching today's data")
        extract_fii_dii_data()
    elif (datetime.now() - FII_DII_LAST_UPDATE).total_seconds() > FII_DII_UPDATE_INTERVAL:
        print(f"\n🔄 FII/DII safety refresh (last: {FII_DII_LAST_UPDATE.strftime('%H:%M')})")
        extract_fii_dii_data()

def check_orb_time_and_process(access_token, live_data):
    global ORB_PROCESSED_TODAY, ORB_LATE_CHECKED
    if not ENABLE_ORB_STRATEGY:
        return
    now          = datetime.now()
    current_time = now.strftime("%H:%M")
    market_920   = now.replace(hour=9, minute=20, second=0, microsecond=0)
    cutoff       = market_920 + timedelta(minutes=ORB_BREAKOUT_WINDOW_MINUTES)

    if current_time < "09:15":
        ORB_PROCESSED_TODAY = False
        ORB_LATE_CHECKED.clear()

    # ── PRIMARY PASS ─────────────────────────────────────────────────────────
    # Old: only ran 09:20-09:25. Bot starting at 09:26 silently skipped ORB.
    # New: run primary pass any time between 09:20 and breakout window close,
    # as long as it hasn't run yet today. One-shot — process_first_candles
    # sets ORB_PROCESSED_TODAY=True so it never fires twice.
    if current_time >= "09:20" and now < cutoff and not ORB_PROCESSED_TODAY:
        if current_time >= "09:25":
            print(f"\n⚠️  ORB: Late start detected ({current_time}). "
                  f"Running primary ORB pass now — {int((now - market_920).total_seconds() / 60)}min "
                  f"into session. Signals use current candle data.")
        process_first_candles(access_token, live_data, late_pass=False)

    # ── LATE VOLUME PASS ─────────────────────────────────────────────────────
    # Retry zero-volume symbols from the primary pass until the window closes.
    elif (ORB_PROCESSED_TODAY
          and ORB_LATE_CHECKED
          and current_time >= "09:25"
          and now < cutoff):
        process_first_candles(access_token, live_data, late_pass=True)

def monitor_orb_breakouts(live_data, trader=None):
    if not ENABLE_ORB_STRATEGY or not ORB_SIGNALS:
        return
    for symbol_key, data in live_data.items():
        try:
            symbol = ISIN_TO_SYMBOL.get(symbol_key, symbol_key)
            if symbol not in ORB_SIGNALS:
                continue
            ltp = data.get('ltp', 0)
            volume = data.get('volume', 0)
            if ltp == 0:
                continue
            breakout = check_orb_breakout(symbol, ltp, volume, data)
            if breakout:
                send_orb_alert(breakout, trader)
        except Exception as e:
            if DEBUG_MODE:
                print(f"ORB monitor error {symbol_key}: {e}")

def print_orb_summary():
    if not ENABLE_ORB_STRATEGY:
        return
    print(f"\n{'='*100}")
    print("📊 ORB STRATEGY SUMMARY")
    print(f"{'='*100}")
    print(f"Total ORB Signals Generated:  {len(ORB_SIGNALS)}")
    print(f"ORB Alerts Triggered:         {len(ORB_ALERTED_STOCKS)}")
    print(f"ORB Orders Placed:            {ORB_ORDER_COUNT}")
    if ORB_SIGNALS:
        very_high = sum(1 for s in ORB_SIGNALS.values() if s['confidence'] == 'VERY_HIGH')
        high = sum(1 for s in ORB_SIGNALS.values() if s['confidence'] == 'HIGH')
        medium = sum(1 for s in ORB_SIGNALS.values() if s['confidence'] == 'MEDIUM')
        print(f"\nConfidence Breakdown:")
        print(f"  VERY HIGH: {very_high}")
        print(f"  HIGH:      {high}")
        print(f"  MEDIUM:    {medium}")
        bullish = sum(1 for s in ORB_SIGNALS.values() if s['is_bullish'])
        bearish = len(ORB_SIGNALS) - bullish
        print(f"\nDirection Breakdown:")
        print(f"  Bullish Setups:  {bullish}")
        print(f"  Bearish Setups:  {bearish}")
    print(f"{'='*100}\n")

def get_token_via_android_oauth() -> str:
    """
    Android-compatible Upstox token refresh using OAuth 2.0 Authorization Code flow.

    Selenium / ChromeDriver does NOT work on Pydroid3 (Android).
    This function replaces that with a simple browser-based flow:

      1. Opens the Upstox login URL in Chrome on your phone.
      2. You log in normally (OTP + PIN) — takes ~30 seconds.
      3. Upstox redirects to https://127.0.0.1/?code=XXXX
      4. You copy that full URL from the Chrome address bar and paste it here.
      5. The bot exchanges the code for an access token via Upstox API.
      6. Token is saved to upstox_token.txt for next run.

    Requires UPSTOX_API_KEY and UPSTOX_API_SECRET to be set above.
    Get them free at https://account.upstox.com/developer/apps

    HOW TO GET API KEY/SECRET (one-time setup, 5 minutes):
      1. Go to https://account.upstox.com/developer/apps
      2. Click "Create New App"
      3. App name: anything (e.g. "MyBot")
      4. Redirect URL: https://127.0.0.1/
      5. Copy API Key (Client ID) and Secret → paste in config above
    """
    import urllib.parse

    if UPSTOX_API_KEY == "YOUR_UPSTOX_API_KEY":
        print("\n❌ UPSTOX_API_KEY not set.")
        print("   Please follow these steps (one-time, 5 minutes):")
        print("   1. Go to https://account.upstox.com/developer/apps")
        print("   2. Create New App → Redirect URL: https://127.0.0.1/")
        print("   3. Copy API Key + Secret → paste in bot config (lines ~45-47)")
        return None

    print("\n" + "=" * 60)
    print("📱 ANDROID OAUTH TOKEN REFRESH")
    print("=" * 60)

    # ── Step 1: Build login URL ───────────────────────────────────────────────
    auth_url = (
        "https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={UPSTOX_API_KEY}"
        f"&redirect_uri={urllib.parse.quote(UPSTOX_REDIRECT_URI, safe='')}"
    )

    print("\n📋 STEP 1: Open this URL in Chrome on your phone:")
    print(f"\n   {auth_url}\n")

    # Try to open it automatically
    try:
        import webbrowser
        webbrowser.open(auth_url)
        print("   ✅ Chrome should have opened. If not, copy-paste the URL above.")
    except Exception:
        print("   ⚠️  Could not auto-open browser. Copy-paste the URL above into Chrome.")

    print("\n📋 STEP 2: Log in to Upstox (OTP + PIN — takes ~30 seconds)")
    print("   After login, Chrome will show a page like:")
    print("   ❌ This site can't be reached  — that's NORMAL!")
    print("   Look at the address bar — it will show something like:")
    print("   https://127.0.0.1/?code=XXXXXXXXXXXXXXXX&state=...")
    print("\n📋 STEP 3: Copy the FULL URL from the address bar and paste it below.")

    # ── Step 2: Wait for user to paste redirect URL ───────────────────────────
    for attempt in range(3):
        try:
            redirect_url = input("\n   Paste the full redirect URL here: ").strip()
        except EOFError:
            print("   ⚠️  Cannot read input in this environment.")
            return None

        if not redirect_url:
            print("   ⚠️  Empty input — try again.")
            continue

        # Parse the auth code from the URL
        try:
            parsed = urllib.parse.urlparse(redirect_url)
            params = urllib.parse.parse_qs(parsed.query)
            auth_code = params.get("code", [None])[0]
        except Exception:
            auth_code = None

        if not auth_code:
            print(f"   ❌ Could not find 'code=' in URL: {redirect_url[:80]}")
            if attempt < 2:
                print("   Make sure you copied the full URL from the address bar.")
            continue

        print(f"\n   ✅ Auth code found: {auth_code[:20]}...")

        # ── Step 3: Exchange code for access token ────────────────────────────
        print("\n🔄 Exchanging auth code for access token...")
        try:
            token_resp = requests.post(
                "https://api.upstox.com/v2/login/authorization/token",
                headers={
                    "accept":       "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "code":          auth_code,
                    "client_id":     UPSTOX_API_KEY,
                    "client_secret": UPSTOX_API_SECRET,
                    "redirect_uri":  UPSTOX_REDIRECT_URI,
                    "grant_type":    "authorization_code",
                },
                timeout=20,
            )
            if token_resp.status_code == 200:
                token_data   = token_resp.json()
                access_token = token_data.get("access_token", "")
                if access_token:
                    print("✅ New access token obtained!")
                    # Save token to file
                    try:
                        with open(UPSTOX_TOKEN_FILE, "w") as f:
                            f.write(access_token)
                        print(f"💾 Token saved to {UPSTOX_TOKEN_FILE}")
                    except Exception as e:
                        print(f"⚠️ Could not save token: {e}")
                    print("\n" + "=" * 60)
                    print("💡 NEXT TIME: Copy this token into HARDCODED_TOKEN in the bot.")
                    print(f"   Token preview: {access_token[:40]}...")
                    print("=" * 60)
                    return access_token
                else:
                    print(f"❌ No access_token in response: {token_resp.text[:200]}")
            else:
                print(f"❌ Token exchange failed ({token_resp.status_code}): {token_resp.text[:200]}")
                if token_resp.status_code == 400:
                    print("   ⚠️  Auth code may have expired (they expire in ~5 minutes).")
                    print("   Run the bot again to get a fresh code.")
                    return None
        except Exception as e:
            print(f"❌ Token exchange error: {e}")

    print("\n❌ OAuth token refresh failed after 3 attempts.")
    return None


def _refresh_upstox_token() -> str:
    """
    Try to refresh the Upstox access token using a saved refresh token.
    Returns new access_token string, or None if refresh fails / no token saved.
    """
    if not os.path.exists(UPSTOX_REFRESH_TOKEN_FILE):
        return None
    try:
        with open(UPSTOX_REFRESH_TOKEN_FILE, "r") as f:
            refresh_token = f.read().strip()
        if not refresh_token:
            return None
        print("🔄 Attempting token refresh via saved refresh_token…")
        resp = requests.post(
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     UPSTOX_API_KEY,
                "client_secret": UPSTOX_API_SECRET,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data         = resp.json()
            access_token = data.get("access_token")
            new_refresh  = data.get("refresh_token", refresh_token)
            if access_token:
                # Persist updated refresh token
                with open(UPSTOX_REFRESH_TOKEN_FILE, "w") as f:
                    f.write(new_refresh)
                with open(UPSTOX_TOKEN_FILE, "w") as f:
                    f.write(access_token)
                print("✅ Token refreshed successfully.")
                return access_token
        print(f"⚠️ Token refresh failed ({resp.status_code}): {resp.text[:100]}")
        return None
    except Exception as exc:
        print(f"⚠️ Token refresh error: {exc}")
        return None


def get_upstox_token():
    """
    Get Upstox access token with smart fallback:
    1. Try hardcoded token if USE_HARDCODED_TOKEN is True
    2. Try saved token from file
    3. Try refresh token (new — no browser needed)
    4. Headless Selenium OAuth (new — fully automated, no manual URL pasting)
    """
    print("=" * 60)
    print("UPSTOX TOKEN MANAGEMENT")
    print("=" * 60)
    print(f"Mobile: {MOBILE_NUMBER}")
    print(f"Email: {EMAIL}")
    print("=" * 60)
    print()

    # STEP 1: Try hardcoded token first if enabled
    if USE_HARDCODED_TOKEN and HARDCODED_TOKEN:
        print("🔑 Step 1: Checking HARDCODED token...")
        print(f"Token preview: {HARDCODED_TOKEN[:30]}...{HARDCODED_TOKEN[-20:]}")

        validation = verify_token(HARDCODED_TOKEN, verbose=True)

        if validation['valid']:
            print("✅ HARDCODED token is VALID - using it!")
            print("=" * 60)
            print()
            try:
                with open(UPSTOX_TOKEN_FILE, 'w') as f:
                    f.write(HARDCODED_TOKEN)
                print(f"💾 Backed up token to {UPSTOX_TOKEN_FILE}")
            except Exception as e:
                print(f"⚠️ Could not backup token: {e}")
            return HARDCODED_TOKEN
        else:
            print(f"❌ HARDCODED token is INVALID: {validation['message']}")
            print("⚠️ Will try other methods...\n")

    # STEP 2: Try saved token from file
    print("🔑 Step 2: Checking SAVED token from file...")
    if os.path.exists(UPSTOX_TOKEN_FILE):
        try:
            with open(UPSTOX_TOKEN_FILE, 'r') as f:
                saved_token = f.read().strip()
            if saved_token:
                print(f"Token preview: {saved_token[:30]}...{saved_token[-20:]}")
                validation = verify_token(saved_token, verbose=True)
                if validation['valid']:
                    print("✅ SAVED token is VALID - using it!")
                    print("=" * 60)
                    print()
                    return saved_token
                else:
                    print(f"❌ SAVED token is INVALID: {validation['message']}")
                    print("⚠️ Will try refresh token...\n")
            else:
                print("⚠️ Token file is empty")
        except Exception as e:
            print(f"❌ Error reading token file: {e}")
    else:
        print(f"⚠️ No saved token file found at {UPSTOX_TOKEN_FILE}")

    # STEP 3: Try refresh token (fast, no browser)
    print("\n🔑 Step 3: Attempting token refresh (no browser needed)...")
    refreshed = _refresh_upstox_token()
    if refreshed:
        validation = verify_token(refreshed, verbose=True)
        if validation['valid']:
            print("✅ Refreshed token is VALID — bot will start now.")
            return refreshed
        else:
            print("❌ Refreshed token failed validation.")

    # STEP 4: Headless Selenium OAuth (fully automated — no manual URL pasting)
    print("\n" + "=" * 60)
    print("🤖 Step 4: HEADLESS SELENIUM OAUTH (fully automated)")
    print("=" * 60)
    print("⚠️  No valid token found. Launching headless Chrome login…")
    print("    Chrome will log in silently — no browser window will appear.")
    print()

    if UPSTOX_API_KEY == "YOUR_UPSTOX_API_KEY":
        print("❌ UPSTOX_API_KEY not set.")
        print("   1. Go to https://account.upstox.com/developer/apps")
        print("   2. Create/open an app → Redirect URL: http://127.0.0.1:8080/")
        print("   3. Paste Client ID into UPSTOX_API_KEY and Client Secret into UPSTOX_API_SECRET")
        return None

    login = UpstoxLogin(
        mobile_number=MOBILE_NUMBER,
        email_address=EMAIL,
        email_password=EMAIL_PASSWORD,
        passcode=PASSCODE,
    )
    new_token = login.perform_oauth_headless()

    if new_token:
        validation = verify_token(new_token, verbose=True)
        if validation['valid']:
            # Persist for next run
            try:
                with open(UPSTOX_TOKEN_FILE, "w") as f:
                    f.write(new_token)
                print(f"💾 Token saved to {UPSTOX_TOKEN_FILE}")
            except Exception:
                pass
            print("✅ Headless token is VALID — bot will start now.")
            return new_token
        else:
            print("❌ Headless token failed validation.")

    print("\n" + "=" * 60)
    print("❌ ALL TOKEN METHODS FAILED")
    print("=" * 60)
    print("\nTo fix:")
    print("  1. Set UPSTOX_API_KEY and UPSTOX_API_SECRET in the bot config")
    print("     (get them at https://account.upstox.com/developer/apps)")
    print("  2. Ensure Chrome/ChromeDriver is installed (pip install webdriver-manager)")
    print("  3. Run the bot — it will log in headlessly and save the token automatically")
    print("  4. Copy the new token into HARDCODED_TOKEN to skip all steps next time")
    return None
def _get_chartink_session():
    """Return a singleton requests.Session configured for ChartInk."""
    global _CHARTINK_SESSION
    if _CHARTINK_SESSION is None:
        _CHARTINK_SESSION = requests.Session()
        _CHARTINK_SESSION.headers.update({
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/x-www-form-urlencoded",
            "origin": "https://chartink.com",
            "referer": "https://chartink.com/stocks-new",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-xsrf-token": CHARTINK_COOKIES.get("XSRF-TOKEN", ""),
        })
    return _CHARTINK_SESSION

def _get_upstox_session(access_token: str) -> requests.Session:
    """
    Return a persistent requests.Session for all Upstox REST calls.
    Re-creates session only when the access_token changes (daily rotation).
    Eliminates DNS + TLS handshake overhead on every request (~200-400 ms saved).
    """
    global _UPSTOX_SESSION, _UPSTOX_SESSION_TOKEN
    token = access_token or ""
    if _UPSTOX_SESSION is None or _UPSTOX_SESSION_TOKEN != token:
        _UPSTOX_SESSION = requests.Session()
        _UPSTOX_SESSION.headers.update({
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        })
        _UPSTOX_SESSION_TOKEN = token
    return _UPSTOX_SESSION

def _fetch_5min_from_chartink(symbol, bars=50):
    """
    PRIMARY: Fetch 5-minute OHLCV candles from ChartInk using robust parsing.

    Args:
        symbol: NSE symbol string e.g. 'RELIANCE'
        bars:   Number of bars to request (max ~200)

    Returns:
        pd.DataFrame with columns [date, open, high, low, close, volume] or None
    """
    try:
        session = _get_chartink_session()

        query = (
            f"select open as 'open', high as 'high', low as 'low', "
            f"close as 'close', volume as 'volume' "
            f"where symbol='{symbol}'"
        )

        payload = {
            "query": query,
            "use_live": "1",
            "limit": str(bars),                # <-- Fixed: was hardcoded "1"
            "size": "200",                      # Max candles per batch
            "widget_id": "-1",
            "end_time": "-1",
            "timeframe": "5 minutes",
            "symbol": symbol,
            "scan_link": "null",
        }

        resp = session.post(
            CHARTINK_BASE_URL,
            cookies=CHARTINK_COOKIES,
            data=payload,
            timeout=20,
        )

        if resp.status_code != 200:
            if DEBUG_MODE:
                print(f"⚠️ ChartInk HTTP {resp.status_code} for {symbol}")
            return None

        data = resp.json()

        # --- Robust parsing (inspired by ChartInkDataExtractor) ---
        if "metaData" not in data or not data["metaData"]:
            if DEBUG_MODE:
                print(f"⚠️ ChartInk missing metaData for {symbol}")
            return None

        # 1. timestamps
        trade_times_ms = data["metaData"][0].get("tradeTimes", [])
        if not trade_times_ms:
            if DEBUG_MODE:
                print(f"⚠️ ChartInk no tradeTimes for {symbol}")
            return None

        # tradeTimes are milliseconds in IST epoch.
        # Convert to IST-aware datetime, then strip tz for naive timestamps.
        timestamps = (
            pd.to_datetime(trade_times_ms, unit="ms", utc=True)
            .tz_convert("Asia/Kolkata")
            .tz_localize(None)
        )

        # 2. column data
        if "groupData" not in data or not data["groupData"]:
            if DEBUG_MODE:
                print(f"⚠️ ChartInk missing groupData for {symbol}")
            return None

        results = data["groupData"][0].get("results", [])
        if not results:
            if DEBUG_MODE:
                print(f"⚠️ ChartInk empty results for {symbol}")
            return None

        # Merge column dicts into one dict of lists
        combined = {}
        for col_dict in results:
            for key, values in col_dict.items():
                norm_key = key.strip().lower()
                # Normalize column names to standard OHLCV
                if norm_key in ("open", "o"):
                    norm_key = "open"
                elif norm_key in ("high", "h"):
                    norm_key = "high"
                elif norm_key in ("low", "l"):
                    norm_key = "low"
                elif norm_key in ("close", "c"):
                    norm_key = "close"
                elif norm_key in ("volume", "v"):
                    norm_key = "volume"
                else:
                    # Skip any extra columns
                    continue

                if norm_key not in combined:
                    combined[norm_key] = []
                combined[norm_key].extend(values)

        # Check we have at least the essential columns
        required = {"open", "high", "low", "close"}
        if not required.issubset(combined.keys()):
            if DEBUG_MODE:
                print(f"⚠️ ChartInk missing required columns for {symbol}: got {list(combined.keys())}")
            return None

        # Ensure all columns have the same length
        col_lens = {k: len(v) for k, v in combined.items()}
        if len(set(col_lens.values())) != 1:
            if DEBUG_MODE:
                print(f"⚠️ ChartInk column length mismatch for {symbol}: {col_lens}")
            return None

        # Build DataFrame
        df = pd.DataFrame(combined)
        df.insert(0, "date", timestamps)

        # Keep only OHLCV and sort
        df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # ── Drop trailing stale bars (ChartInk ~5–6 min pipeline delay) ──────
        # bar[-1] = still-forming current bar with stale data → always drop
        # bar[-2] = last "complete" bar but often also delayed → drop too
        # The real-time candle builder fills these from live Upstox LTP ticks.
        if CK_BARS_TO_DROP > 0 and len(df) > CK_BARS_TO_DROP + 5:
            dropped_bars = df.tail(CK_BARS_TO_DROP)["date"].dt.strftime("%H:%M").tolist()
            df = df.iloc[:-CK_BARS_TO_DROP].reset_index(drop=True)
            if DEBUG_MODE:
                print(f"✂️  {symbol}: Dropped {CK_BARS_TO_DROP} stale CK bars "
                      f"({', '.join(dropped_bars)}) — RT builder will fill gap")

        # ── Lag warning: how old is the last bar we kept? ─────────────────────
        if len(df) > 0 and DEBUG_MODE:
            last_ts = df["date"].iloc[-1]
            if hasattr(last_ts, "to_pydatetime"):
                last_ts = last_ts.to_pydatetime().replace(tzinfo=None)
            lag_min = (datetime.now() - last_ts).total_seconds() / 60
            if lag_min > CK_LAG_WARN_MIN:
                print(f"⚠️  CK LAG {symbol}: last kept bar={last_ts.strftime('%H:%M')}, "
                      f"lag={lag_min:.1f} min — RT builder must cover this gap")

        if DEBUG_MODE:
            print(f"✅ ChartInk 5min: {symbol} → {len(df)} bars (after lag trim)")

        return df if len(df) >= 5 else None

    except requests.exceptions.Timeout:
        if DEBUG_MODE:
            print(f"⏱️ ChartInk timeout for {symbol}")
        return None
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ ChartInk error for {symbol}: {e}")
        return None

def _record_5min_failure(instrument_key):
    """Track consecutive Upstox 5min data failures and blacklist after MAX_5MIN_FAILURES."""
    global FAST_TRADE_5MIN_BLACKLIST
    count = FAST_TRADE_5MIN_FAILURES.get(instrument_key, 0) + 1
    FAST_TRADE_5MIN_FAILURES[instrument_key] = count
    if count >= MAX_5MIN_FAILURES:
        if instrument_key not in FAST_TRADE_5MIN_BLACKLIST:
            FAST_TRADE_5MIN_BLACKLIST.add(instrument_key)
            if DEBUG_MODE:
                print(f"🚫 Blacklisted {instrument_key} for Upstox 5min data (failed {count} times)")


# ============ REAL-TIME 5MIN CANDLE BUILDER ============

def get_current_5min_slot():
    """Get the start time of the current 5-minute candle slot"""
    now = datetime.now()
    minute = (now.minute // 5) * 5
    return now.replace(minute=minute, second=0, microsecond=0)

def update_realtime_candle(symbol, ltp, volume):
    """
    Feed a new tick into the real-time 5min candle builder.
    Call this every time you get a live price update.
    """
    if not ltp or ltp <= 0:
        return

    current_slot = get_current_5min_slot()

    with CANDLE_BUILDER_LOCK:
        if symbol not in CURRENT_CANDLE:
            CURRENT_CANDLE[symbol] = {
                'open': ltp,
                'high': ltp,
                'low': ltp,
                'close': ltp,
                'volume': volume or 0,
                'candle_start': current_slot
            }
            return

        candle = CURRENT_CANDLE[symbol]

        if current_slot > candle['candle_start']:
            # Close the previous candle and save it
            completed = {
                'date': candle['candle_start'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            }
            REALTIME_CANDLES[symbol].append(completed)

            # Keep only last 100 candles to save memory
            if len(REALTIME_CANDLES[symbol]) > 100:
                REALTIME_CANDLES[symbol] = REALTIME_CANDLES[symbol][-100:]

            # Start new candle
            CURRENT_CANDLE[symbol] = {
                'open': ltp,
                'high': ltp,
                'low': ltp,
                'close': ltp,
                'volume': volume or 0,
                'candle_start': current_slot
            }
        else:
            # Update current candle
            candle['high'] = max(candle['high'], ltp)
            candle['low'] = min(candle['low'], ltp)
            candle['close'] = ltp
            # Volume from Upstox is cumulative day volume, store latest value
            candle['volume'] = volume or candle['volume']

def get_realtime_5min_df(symbol, min_bars=20):
    """
    Get completed 5min candles for a symbol as DataFrame.
    Returns None if insufficient data.
    """
    with CANDLE_BUILDER_LOCK:
        candles = REALTIME_CANDLES.get(symbol, [])

        if len(candles) < min_bars:
            if DEBUG_MODE:
                print(f"⚠️ {symbol}: Only {len(candles)} real-time candles built so far "
                      f"(need {min_bars})")
            return None

        df = pd.DataFrame(candles)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Convert cumulative volume to per-bar volume
        df['volume'] = df['volume'].diff().fillna(df['volume'].iloc[0])
        df['volume'] = df['volume'].clip(lower=0)

        return df


def _fetch_5min_upstox_intraday(access_token, instrument_key, timeframe="5minute"):
    """Upstox intraday endpoint — supports 5minute or 15minute."""
    url     = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{timeframe}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    try:
        resp = _get_upstox_session(access_token).get(url, timeout=15)
        if resp.status_code != 200:
            return None, headers
        data = resp.json()
        candles = data.get("data", {}).get("candles", [])
        if not candles or len(candles) < 5:
            return None, headers
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume", "oi"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df, headers
    except Exception:
        return None, headers


def _fetch_5min_upstox_historical(access_token, instrument_key, headers, timeframe="5minute"):
    """Upstox historical date-range endpoint — supports 5minute or 15minute."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{timeframe}/{to_date}/{from_date}"
    try:
        resp = _get_upstox_session(access_token).get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        candles = data.get("data", {}).get("candles", [])
        if not candles or len(candles) < 5:
            return None
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume", "oi"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def _get_chartink_hist_base(symbol, bars=100):
    """
    Fetch (or return cached) ChartInk historical 5-min OHLCV for `symbol`.

    Called ONCE on first use per symbol; result is cached for _CK_HIST_CACHE_TTL
    seconds.  Gives ~50-100 clean bars spanning today + yesterday with no warmup
    period — available from 09:15 on day open.

    Returns DataFrame[date, open, high, low, close, volume] or None.
    """
    with _CK_HIST_CACHE_LOCK:
        now = datetime.now()
        cached_ts = _CK_HIST_CACHE_TS.get(symbol)
        if (
            symbol in _CK_HIST_CACHE
            and cached_ts is not None
            and (now - cached_ts).total_seconds() < _CK_HIST_CACHE_TTL
        ):
            return _CK_HIST_CACHE[symbol]

        # Fetch fresh
        df = _fetch_5min_from_chartink(symbol, bars=bars)
        if df is not None and len(df) >= 5:
            _CK_HIST_CACHE[symbol] = df
            _CK_HIST_CACHE_TS[symbol] = now
            if DEBUG_MODE:
                print(f"💾 CK hist cached: {symbol} → {len(df)} bars")
        else:
            # Keep previous cache (stale) rather than returning None
            if symbol not in _CK_HIST_CACHE:
                _CK_HIST_CACHE[symbol] = None
            if DEBUG_MODE:
                print(f"⚠️ CK hist fetch failed for {symbol} — using stale cache")

        return _CK_HIST_CACHE.get(symbol)


def _merge_hist_and_realtime(hist_df, symbol):
    """
    Merge ChartInk historical base with real-time LTP data.

    Layers:
      1. ChartInk hist base  — clean OHLCV up to ~5-6 min ago
      2. REALTIME_CANDLES    — completed 5-min bars built from live LTP ticks
      3. CURRENT_CANDLE      — the still-open bar right now (appended as synthetic bar)

    Deduplication: real-time data wins on any overlapping 5-min slot.
    """
    frames = [hist_df]

    with CANDLE_BUILDER_LOCK:
        rt_candles = list(REALTIME_CANDLES.get(symbol, []))
        current    = dict(CURRENT_CANDLE.get(symbol, {}))

    if rt_candles:
        rt_df = pd.DataFrame(rt_candles)
        rt_df["date"] = pd.to_datetime(rt_df["date"])
        frames.append(rt_df)

    if current and current.get("open") is not None:
        cur_df = pd.DataFrame([{
            "date":   current["candle_start"],
            "open":   current["open"],
            "high":   current["high"],
            "low":    current["low"],
            "close":  current["close"],
            "volume": current.get("volume", 0),
        }])
        cur_df["date"] = pd.to_datetime(cur_df["date"])
        frames.append(cur_df)

    if len(frames) == 1:
        # No real-time data yet — the CK lag gap is uncovered, warn loudly
        if len(hist_df) > 0:
            last_ts = hist_df["date"].iloc[-1]
            if hasattr(last_ts, "to_pydatetime"):
                last_ts = last_ts.to_pydatetime().replace(tzinfo=None)
            lag_min = (datetime.now() - last_ts).total_seconds() / 60
            if lag_min > CK_LAG_WARN_MIN:
                print(f"⚠️  {symbol}: CK hist ends at {last_ts.strftime('%H:%M')} "
                      f"({lag_min:.1f} min ago) and NO real-time bars yet — "
                      f"signals will use stale data until RT builder catches up!")
        return hist_df

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # Strip timezone if mixed
    if combined["date"].dt.tz is not None:
        combined["date"] = combined["date"].dt.tz_localize(None)

    # Deduplicate by 5-min slot: keep last (real-time wins over historical)
    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return combined


def fetch_5min_candle_data(access_token, instrument_key, bars=100, symbol=None):
    """
    Fetch 5-minute candle data for fast trading.

    HYBRID SOURCE PRIORITY:
      1. ChartInk historical base  +  Upstox real-time LTP  (HYBRID — PRIMARY)
         ChartInk gives clean OHLCV history from the previous session onward.
         Real-time LTP ticks (from update_realtime_candle) fill the gap to now.
         Result is a complete dataset from the very first scan at 09:15.

      2. Pure real-time candles only  (fallback if ChartInk fails, needs 20 bars)

      3. Upstox intraday endpoint  (LAST RESORT 1 — needs NSE_FO key)

      4. Upstox historical endpoint  (LAST RESORT 2)

    Args:
        access_token:   Upstox Bearer token
        instrument_key: NSE_EQ instrument_key (for blacklist + FO lookup)
        bars:           Historical bars to request from ChartInk (default 100)
        symbol:         NSE symbol string e.g. 'RELIANCE'
    """
    if symbol is None:
        symbol = ISIN_TO_SYMBOL.get(instrument_key, "")

    # ── SOURCE 1: ChartInk historical + real-time LTP (HYBRID) ──────────────
    if symbol:
        hist_df = _get_chartink_hist_base(symbol, bars=bars)
        if hist_df is not None and len(hist_df) >= 20:
            merged = _merge_hist_and_realtime(hist_df, symbol)
            FAST_TRADE_5MIN_FAILURES.pop(instrument_key, None)
            if DEBUG_MODE:
                rt_bars  = len(REALTIME_CANDLES.get(symbol, []))
                last_ts  = merged["date"].iloc[-1]
                if hasattr(last_ts, "to_pydatetime"):
                    last_ts = last_ts.to_pydatetime().replace(tzinfo=None)
                lag_min  = (datetime.now() - last_ts).total_seconds() / 60
                print(
                    f"✅ Hybrid 5min: {symbol} → {len(merged)} bars "
                    f"(CK hist={len(hist_df)}, RT={rt_bars}) "
                    f"| last bar={last_ts.strftime('%H:%M')} lag={lag_min:.1f}min"
                )
            return merged

    # ── SOURCE 2: Pure real-time candles (ChartInk unavailable) ─────────────
    if symbol:
        df = get_realtime_5min_df(symbol, min_bars=20)
        if df is not None:
            FAST_TRADE_5MIN_FAILURES.pop(instrument_key, None)
            if DEBUG_MODE:
                print(f"✅ Real-time only 5min: {symbol} → {len(df)} bars")
            return df

    # ── SOURCE 3/4: Upstox endpoints (last resort) ────────────────────────────
    if instrument_key in FAST_TRADE_5MIN_BLACKLIST:
        if DEBUG_MODE:
            print(f"🚫 {instrument_key} blacklisted for Upstox 5min — skipping")
        return None

    fo_key = SYMBOL_TO_FO_KEY.get(symbol, instrument_key) if symbol else instrument_key

    df, headers = _fetch_5min_upstox_intraday(access_token, fo_key, timeframe="5minute")
    if df is not None:
        FAST_TRADE_5MIN_FAILURES.pop(instrument_key, None)
        if DEBUG_MODE:
            print(f"✅ Upstox intraday 5min: {fo_key} → {len(df)} bars")
        return df

    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    df = _fetch_5min_upstox_historical(access_token, fo_key, headers, timeframe="5minute")
    if df is not None:
        FAST_TRADE_5MIN_FAILURES.pop(instrument_key, None)
        if DEBUG_MODE:
            print(f"✅ Upstox historical 5min: {fo_key} → {len(df)} bars")
        return df

    _record_5min_failure(instrument_key)
    return None


def fetch_15min_candle_data(access_token, instrument_key, symbol=None):
    """
    Fetch 15-minute candle data for SQUEEZE (LONG) signal detection.

    Source priority:
      1. ChartInk historical (tf=15) — primary, clean OHLCV
      2. Upstox intraday 15minute endpoint — fallback
      3. Upstox historical 15minute endpoint — last resort
      4. Resample from 5min data — emergency fallback

    Returns DataFrame with columns: date, open, high, low, close, volume
    or None if insufficient data.
    """
    if symbol is None:
        symbol = ISIN_TO_SYMBOL.get(instrument_key, "")

    # ── SOURCE 1: ChartInk 15min ─────────────────────────────────────────────
    if symbol:
        df15 = _fetch_15min_from_chartink(symbol, bars=60)
        if df15 is not None and len(df15) >= 10:
            if DEBUG_MODE:
                last_ts  = df15["date"].iloc[-1]
                lag_min  = (datetime.now() - last_ts).total_seconds() / 60
                print(f"✅ ChartInk 15min: {symbol} → {len(df15)} bars "
                      f"| last bar={last_ts.strftime('%H:%M')} lag={lag_min:.1f}min")
            return df15

    # ── SOURCE 2: Upstox intraday 15min ─────────────────────────────────────
    fo_key = SYMBOL_TO_FO_KEY.get(symbol, instrument_key) if symbol else instrument_key
    df15, headers = _fetch_5min_upstox_intraday(access_token, fo_key, timeframe="15minute")
    if df15 is not None and len(df15) >= 10:
        if DEBUG_MODE:
            print(f"✅ Upstox intraday 15min: {fo_key} → {len(df15)} bars")
        return df15

    # ── SOURCE 3: Upstox historical 15min ────────────────────────────────────
    df15 = _fetch_5min_upstox_historical(access_token, fo_key, {}, timeframe="15minute")
    if df15 is not None and len(df15) >= 10:
        if DEBUG_MODE:
            print(f"✅ Upstox historical 15min: {fo_key} → {len(df15)} bars")
        return df15

    # ── SOURCE 4: Resample 5min → 15min (emergency fallback) ─────────────────
    df5 = fetch_5min_candle_data(access_token, instrument_key, bars=100, symbol=symbol)
    if df5 is not None and len(df5) >= 30:
        try:
            df5 = df5.tail(100).copy()   # limit input before resample — saves CPU
            df5['date'] = pd.to_datetime(df5['date'])
            df5 = df5.set_index('date')
            df15 = df5.resample('15min').agg({
                'open':   'first',
                'high':   'max',
                'low':    'min',
                'close':  'last',
                'volume': 'sum'
            }).dropna().reset_index()
            if len(df15) >= 10:
                if DEBUG_MODE:
                    print(f"✅ Resampled 5→15min: {symbol} → {len(df15)} bars")
                return df15
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ 15min resample failed for {symbol}: {e}")

    if DEBUG_MODE:
        print(f"⚠️ {symbol}: No 15min data available")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# INTRADAY CANDLE CACHE WRAPPERS
# Wrap fetch_5min_candle_data / fetch_15min_candle_data with a short TTL cache
# so the same data is never fetched twice within a single 30s scan cycle.
# ══════════════════════════════════════════════════════════════════════════════

def fetch_5min_cached(access_token: str, instrument_key: str,
                      bars: int = 50, symbol: str = None) -> 'pd.DataFrame | None':
    """
    Cached wrapper around fetch_5min_candle_data.
    Returns a cached DataFrame if one was fetched within _5MIN_CACHE_TTL_S seconds.
    Thread-safe: uses _INTRADAY_CACHE_LOCK for reads/writes.
    """
    if symbol is None:
        symbol = ISIN_TO_SYMBOL.get(instrument_key, "")
    key = symbol or instrument_key
    now = datetime.now()

    with _INTRADAY_CACHE_LOCK:
        entry = _5MIN_CACHE.get(key)
        if entry is not None:
            age = (now - entry['fetched_at']).total_seconds()
            if age < _5MIN_CACHE_TTL_S and entry.get('requested_bars', 0) >= bars:
                return entry['df']   # cache hit — no network call

    # Cache miss — fetch
    df = fetch_5min_candle_data(access_token, instrument_key, bars=bars, symbol=symbol)

    with _INTRADAY_CACHE_LOCK:
        _5MIN_CACHE[key] = {'df': df, 'fetched_at': now, 'requested_bars': bars}

    return df


def fetch_15min_cached(access_token: str, instrument_key: str,
                       symbol: str = None) -> 'pd.DataFrame | None':
    """
    Cached wrapper around fetch_15min_candle_data.
    Returns a cached DataFrame if one was fetched within _15MIN_CACHE_TTL_S seconds.
    Thread-safe: uses _INTRADAY_CACHE_LOCK for reads/writes.
    """
    if symbol is None:
        symbol = ISIN_TO_SYMBOL.get(instrument_key, "")
    key = symbol or instrument_key
    now = datetime.now()

    with _INTRADAY_CACHE_LOCK:
        entry = _15MIN_CACHE.get(key)
        if entry is not None:
            age = (now - entry['fetched_at']).total_seconds()
            if age < _15MIN_CACHE_TTL_S:
                return entry['df']

    df = fetch_15min_candle_data(access_token, instrument_key, symbol=symbol)

    with _INTRADAY_CACHE_LOCK:
        _15MIN_CACHE[key] = {'df': df, 'fetched_at': now}

    return df


def clear_intraday_cache():
    """Flush both intraday caches. Called once per scan cycle by the main loop."""
    with _INTRADAY_CACHE_LOCK:
        _5MIN_CACHE.clear()
        _15MIN_CACHE.clear()


def _fetch_15min_from_chartink(symbol: str, bars: int = 60) -> 'pd.DataFrame | None':
    """
    Fetch 15-minute historical OHLCV from ChartInk.
    Uses tf=15 parameter — same endpoint as 5min but with different timeframe.
    """
    try:
        url = f"{CHARTINK_BASE_URL}/chart_data/"
        params = {
            "symbol":    symbol,
            "tf":        "15",        # 15-minute timeframe
            "type":      "stock",
            "exchange":  "NSE",
            "bars":      str(bars),
        }
        ck_sess = _get_chartink_session()
        resp = ck_sess.get(
            url, params=params, cookies=CHARTINK_COOKIES,
            headers={"Referer": "https://chartink.com/"},
            timeout=15
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # ChartInk returns arrays: t (timestamps), o, h, l, c, v
        t = data.get("t", [])
        if not t or len(t) < 5:
            return None
        df = pd.DataFrame({
            "date":   pd.to_datetime(t, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None),
            "open":   data.get("o", []),
            "high":   data.get("h", []),
            "low":    data.get("l", []),
            "close":  data.get("c", []),
            "volume": data.get("v", []),
        })
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None
def get_option_premium_with_fallback(trader, contract, spot_price, max_retries):
    """Get option premium with retry and intelligent fallback."""
    premium = None
    instrument_key = contract["instrument_key"]
    
    # Try to get real LTP
    for attempt in range(max_retries):
        try:
            premium = trader.get_ltp(instrument_key, max_retries=1)
            if premium and premium > 0:
                return premium, False  # Real LTP
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ Premium fetch error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1.5)
    
    # Fallback: Estimate premium
    print(f"⚠️ Using estimated premium for {contract.get('trading_symbol')}")
    
    strike = contract["strike_price"]
    option_type = contract["instrument_type"]
    
    # Intrinsic value
    if option_type == 'CE':
        intrinsic = max(0, spot_price - strike)
    else:  # PE
        intrinsic = max(0, strike - spot_price)
    
    # Time value estimation based on moneyness and volatility
    moneyness = abs(strike - spot_price) / spot_price
    
    # ATM options have more time value
    if moneyness < 0.02:  # Very near ATM
        time_value_pct = 0.05  # 5% of spot
    elif moneyness < 0.05:  # Near ATM
        time_value_pct = 0.03  # 3% of spot
    elif moneyness < 0.10:  # Slightly OTM
        time_value_pct = 0.02  # 2% of spot
    else:  # Deep OTM
        time_value_pct = 0.01  # 1% of spot
    
    time_value = spot_price * time_value_pct
    estimated_premium = intrinsic + time_value
    
    # Minimum floor
    estimated_premium = max(estimated_premium, spot_price * 0.015)  # At least 1.5%
    
    return estimated_premium, True  # Estimated premium

def validate_premium(premium, spot_price, symbol):
    """Validate premium is within reasonable bounds."""
    premium_as_pct = (premium / spot_price) * 100
    
    # Check minimum
    if premium < OPTION_PREMIUM_MIN_THRESHOLD:
        print(f"⚠️ Premium too low (₹{premium:.2f}, {premium_as_pct:.1f}%) for {symbol}")
        return False
    
    # Check maximum (unusually high premium might indicate error)
    if premium > OPTION_PREMIUM_MAX_THRESHOLD:
        print(f"⚠️ Premium too high (₹{premium:.2f}, {premium_as_pct:.1f}%) for {symbol}")
        return False
    
    # Premium should typically be 1-20% of spot for equity options
    if premium_as_pct > 30:
        print(f"⚠️ Premium suspiciously high ({premium_as_pct:.1f}%) for {symbol}")
        return False
    
    return True

def get_all_fno_equities(access_token):
    print("📥 Downloading F&O stock list...")
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        df = pd.read_csv(url, compression='gzip')
        fo = df[df['exchange']=='NSE_FO']
        fo_symbols = fo['tradingsymbol'].str.replace(r'\d{2}[A-Z]{3}\d{2,4}.*','', regex=True).str.strip().unique()
        fo_symbols = set([s for s in fo_symbols if s])
        eq = df[(df['exchange']=='NSE_EQ') & (df['tradingsymbol'].isin(fo_symbols))].copy()
        eq = eq.drop_duplicates(subset=['tradingsymbol'])
        keys = eq['instrument_key'].tolist()
        sym = dict(zip(eq['instrument_key'], eq['tradingsymbol']))
        print(f"✅ Found {len(keys)} F&O stocks\n")
        return keys, sym
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], {}

def fetch_historical_ohlc(access_token, instrument_key, target_date):
    """Fetch OHLC data for a specific date using v2 API.
    Retries once on 429 with a 5-second back-off."""
    date_str = (target_date.strftime('%Y-%m-%d')
                if isinstance(target_date, datetime) or hasattr(target_date, 'strftime')
                else str(target_date))
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{date_str}/{date_str}"
    for attempt in range(2):   # 1 normal attempt + 1 retry on 429
        try:
            resp = _get_upstox_session(access_token).get(url, timeout=15)
            if resp.status_code == 200:
                data    = resp.json()
                candles = data.get("data", {}).get("candles", [])
                if not candles:
                    return None
                candle = candles[0]
                return {
                    "date":   target_date,
                    "open":   candle[1],
                    "high":   candle[2],
                    "low":    candle[3],
                    "close":  candle[4],
                    "volume": candle[5],
                }
            elif resp.status_code == 429:
                if attempt == 0:
                    if DEBUG_MODE:
                        print(f"⚠️ fetch_historical_ohlc 429 for {instrument_key} — waiting 5s")
                    time.sleep(5)
                    continue   # retry once
                return None
            else:
                return None
        except Exception as e:
            if DEBUG_MODE:
                print(f" OHLC fetch exception: {e}")
            return None
    return None

def fetch_volume_history(access_token, instrument_key, end_date, days=40):
    """Fetch volume history using cached candle data"""
    # Use cached candle fetching instead of direct API call
    symbol = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key.split(':')[-1]
    df = get_cached_or_fetch_candles(access_token, symbol, instrument_key)
    
    if df is None or len(df) < days:
        return None
    
    # Filter to requested date range
    df = df[df['date'] <= pd.Timestamp(end_date)]
    df = df.tail(days)
    
    return df

def get_live_prices_batch(access_token, instrument_keys):
    """Fetch live prices using ISIN keys and map responses back"""
    if not instrument_keys:
        return {}
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    url = "https://api.upstox.com/v2/market-quote/quotes"
    results = {}
    
    for i in range(0, len(instrument_keys), MAX_INSTRUMENTS_PER_BATCH):
        chunk = instrument_keys[i:i+MAX_INSTRUMENTS_PER_BATCH]
        params = [('instrument_key', key) for key in chunk]
        try:
            response = _get_upstox_session(access_token).get(url, params=params, timeout=30)
            if DEBUG_MODE and i == 0:
                print(f"📡 Batch API Call #{i//MAX_INSTRUMENTS_PER_BATCH + 1}: Status {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    for response_key, quote in data['data'].items():
                        nk_response = norm_key(response_key)
                        if nk_response in R3_LEVELS:
                            isin_key = nk_response
                        else:
                            symbol = response_key.split('|')[-1] if '|' in response_key else response_key.split(':')[-1]
                            isin_key = SYMBOL_TO_ISIN.get(symbol)
                            if not isin_key:
                                for stored_key, stored_info in R3_LEVELS.items():
                                    if stored_info['symbol'] == symbol:
                                        isin_key = stored_key
                                        break
                        if isin_key and isin_key in R3_LEVELS:
                            ohlc_data = quote.get('ohlc', {})
                            results[isin_key] = {
                                'ltp': quote.get('last_price'),
                                'high': ohlc_data.get('high'),
                                'low': ohlc_data.get('low'),
                                'open': ohlc_data.get('open'),
                                'close': ohlc_data.get('close'),
                                'volume': quote.get('volume'),
                                'timestamp': datetime.now()
                            }
            elif response.status_code == 429:
                print(" ⚡ Rate limit hit, waiting...")
                time.sleep(5)
                continue
        except Exception as e:
            if DEBUG_MODE:
                print(f" ❌ Batch fetch error: {e}")
        if i + MAX_INSTRUMENTS_PER_BATCH < len(instrument_keys):
            time.sleep(0.5)
    return results

# ========== CALCULATION FUNCTIONS ==========
def calc_r3(h, l, c):
    p = (h + l + c) / 3.0
    r3 = p + 2*(h - l)
    return p, r3

def calc_s3(h, l, c):
    """Calculate S3 (Support Level 3)"""
    p = (h + l + c) / 3.0
    s3 = p - 2*(h - l)
    return p, s3

def _calculate_rsi_series(close_series, period=14):
    """Return an RSI series for the given close prices."""
    close = pd.to_numeric(pd.Series(close_series), errors='coerce').reset_index(drop=True)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    flat_mask = (avg_gain.fillna(0) == 0) & (avg_loss.fillna(0) == 0)
    rsi = rsi.mask(flat_mask, 50.0)
    rsi = rsi.mask((avg_loss.fillna(0) == 0) & (avg_gain.fillna(0) > 0), 100.0)
    rsi = rsi.mask((avg_gain.fillna(0) == 0) & (avg_loss.fillna(0) > 0), 0.0)
    return rsi

def calculate_rsi(data, period=14):
    """Return the latest RSI value for a DataFrame or close-price series."""
    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        if 'close' not in data.columns:
            return None
        close = data['close']
    else:
        close = pd.Series(data)

    close = pd.to_numeric(close, errors='coerce').dropna().reset_index(drop=True)
    if len(close) < period + 1:
        return None

    rsi = _calculate_rsi_series(close, period=period)
    latest = rsi.iloc[-1]
    return float(latest) if pd.notna(latest) else None

def _normalize_session_volumes(day_df):
    """
    Normalize a day's volume series into per-bar volumes.
    ChartInk history is already per-bar, while real-time builder bars are cumulative.
    This helper converts any cumulative tail back to per-bar volumes so VWAP
    stays usable intraday.
    """
    volumes = pd.to_numeric(day_df['volume'], errors='coerce').fillna(0).clip(lower=0).astype(float).reset_index(drop=True)
    if len(volumes) <= 1:
        return volumes

    non_zero = volumes[volumes > 0]
    median_vol = float(non_zero.median()) if not non_zero.empty else 0.0
    diffs = volumes.diff().fillna(0)
    non_decreasing_ratio = float((diffs >= 0).mean())

    if median_vol > 0 and non_decreasing_ratio >= 0.85 and volumes.iloc[-1] > median_vol * 4:
        return volumes.diff().fillna(volumes.iloc[0]).clip(lower=0)

    split_idx = None
    for start in range(1, len(volumes)):
        prefix = volumes.iloc[:start]
        suffix = volumes.iloc[start:]
        prefix_non_zero = prefix[prefix > 0]
        prefix_median = float(prefix_non_zero.median()) if not prefix_non_zero.empty else median_vol
        suffix_diffs = suffix.diff().fillna(0)

        if float((suffix_diffs >= 0).mean()) < 0.85:
            continue
        if suffix.iloc[0] <= max(prefix_median * 3, median_vol * 2, 1.0):
            continue
        if prefix.sum() > 0 and suffix.iloc[0] < prefix.sum() * 0.20:
            continue
        split_idx = start
        break

    if split_idx is None:
        return volumes

    normalized = volumes.copy()
    prior_sum = float(volumes.iloc[:split_idx].sum())
    normalized.iloc[split_idx] = max(volumes.iloc[split_idx] - prior_sum, 0.0)
    if split_idx + 1 < len(volumes):
        normalized.iloc[split_idx + 1:] = volumes.iloc[split_idx + 1:].diff().clip(lower=0)
    return normalized.clip(lower=0)

def build_intraday_indicator_frame(df):
    """Build a clean intraday OHLCV frame with EMA, VWAP, and RSI helpers."""
    if df is None or df.empty:
        return None

    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    out.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    if out.empty:
        return None

    out['session_date'] = out['date'].dt.date
    out['session_volume'] = 0.0
    for session_date, idx in out.groupby('session_date').groups.items():
        idx = list(idx)
        out.loc[idx, 'session_volume'] = _normalize_session_volumes(out.loc[idx, ['volume']]).values

    out['ema_200'] = out['close'].ewm(span=TREND_FILTER_EMA_PERIOD, adjust=False).mean()
    out['ema_trail'] = out['close'].ewm(span=PULLBACK_TRAIL_EMA_PERIOD, adjust=False).mean()
    out['rsi_2'] = _calculate_rsi_series(out['close'], period=PULLBACK_RSI_PERIOD)

    typical_price = (out['high'] + out['low'] + out['close']) / 3.0
    cum_tpv = (typical_price * out['session_volume']).groupby(out['session_date']).cumsum()
    cum_vol = out['session_volume'].groupby(out['session_date']).cumsum()
    out['vwap'] = np.where(cum_vol > 0, cum_tpv / cum_vol, out['close'])

    candle_range = (out['high'] - out['low']).replace(0, np.nan)
    out['body_size'] = (out['close'] - out['open']).abs()
    out['body_pct'] = np.where(out['open'].abs() > 0, out['body_size'] / out['open'] * 100, 0.0)
    out['is_green'] = out['close'] > out['open']
    out['is_red'] = out['close'] < out['open']
    out['upper_wick_pct'] = np.where(
        candle_range > 0,
        (out['high'] - np.maximum(out['open'], out['close'])) / candle_range * 100,
        0.0
    )
    out['lower_wick_pct'] = np.where(
        candle_range > 0,
        (np.minimum(out['open'], out['close']) - out['low']) / candle_range * 100,
        0.0
    )
    return out.reset_index(drop=True)

def _count_big_green_candles(df_slice):
    if df_slice is None or df_slice.empty:
        return 0
    mask = (
        (df_slice['close'] > df_slice['open'])
        & (df_slice['body_pct'] >= PULLBACK_BIG_GREEN_BODY_PERCENT)
    )
    return int(mask.sum())

def _is_strong_bullish_candle(bar):
    return (
        bar is not None
        and bar.get('close', 0) > bar.get('open', 0)
        and bar.get('body_pct', 0) >= PULLBACK_STRONG_BULL_BODY_PERCENT
        and bar.get('upper_wick_pct', 100) <= 45
    )

def _is_bullish_engulfing(current_bar, prev_bar):
    if current_bar is None or prev_bar is None:
        return False
    if current_bar.get('close', 0) <= current_bar.get('open', 0):
        return False
    if prev_bar.get('close', 0) >= prev_bar.get('open', 0):
        return False
    if current_bar.get('body_pct', 0) < PULLBACK_STRONG_BULL_BODY_PERCENT:
        return False
    return (
        current_bar.get('open', 0) <= prev_bar.get('close', 0)
        and current_bar.get('close', 0) >= prev_bar.get('open', 0)
    )

def build_pullback_ce_signal(access_token, instrument_key, info, live):
    """
    Build the refined CE setup:
      1. Trend filter: price above VWAP and EMA200, EMA sloping up
      2. Pullback: RSI(2) on a recent completed candle below threshold
      3. Trigger: previous-candle/minor-resistance break or strong bullish engulfing
      4. Avoid: overbought RSI, 3 stretched green candles, price too far above VWAP
    """
    if not ENABLE_PULLBACK_CE_STRATEGY:
        return None

    symbol = info['symbol']
    current_price = live.get('ltp')
    day_open = live.get('open')

    if not current_price or current_price <= 0:
        return None
    if day_open and current_price <= day_open:
        return None
    if info.get('yesterday_close') and current_price <= info['yesterday_close']:
        return None

    bars_needed = max(TREND_FILTER_EMA_PERIOD + 25, 240)
    df5 = fetch_5min_cached(access_token, instrument_key, bars=bars_needed, symbol=symbol)
    if df5 is None or len(df5) < TREND_FILTER_EMA_PERIOD:
        return None

    df_ind = build_intraday_indicator_frame(df5)
    if df_ind is None or len(df_ind) < TREND_FILTER_EMA_PERIOD:
        return None

    today = datetime.now().date()
    day_df = df_ind[df_ind['session_date'] == today].copy().reset_index(drop=True)
    if len(day_df) < max(PULLBACK_LOOKBACK_BARS + 3, 8):
        return None

    completed = day_df.iloc[:-1].copy().reset_index(drop=True)
    if len(completed) < max(PULLBACK_LOOKBACK_BARS + 2, 6):
        return None

    current_bar = day_df.iloc[-1].copy()
    current_open = float(current_bar['open']) if pd.notna(current_bar['open']) else current_price
    current_high = float(current_bar['high']) if pd.notna(current_bar['high']) else current_price
    current_low = float(current_bar['low']) if pd.notna(current_bar['low']) else current_price

    current_bar['open'] = current_open
    current_bar['close'] = float(current_price)
    current_bar['high'] = max(current_high, float(current_price))
    current_bar['low'] = min(current_low, float(current_price))
    current_bar['body_size'] = abs(current_bar['close'] - current_bar['open'])
    current_bar['body_pct'] = (current_bar['body_size'] / current_bar['open'] * 100) if current_bar['open'] else 0.0
    candle_range = current_bar['high'] - current_bar['low']
    current_bar['is_green'] = current_bar['close'] > current_bar['open']
    current_bar['upper_wick_pct'] = (
        ((current_bar['high'] - max(current_bar['open'], current_bar['close'])) / candle_range) * 100
        if candle_range > 0 else 0.0
    )

    close_for_ema = df_ind['close'].copy()
    close_for_ema.iloc[-1] = current_price
    live_ema_200 = float(close_for_ema.ewm(span=TREND_FILTER_EMA_PERIOD, adjust=False).mean().iloc[-1])
    prev_ema_200 = float(df_ind['ema_200'].iloc[-2]) if len(df_ind) >= 2 and pd.notna(df_ind['ema_200'].iloc[-2]) else live_ema_200

    live_vwap = float(current_bar['vwap']) if pd.notna(current_bar['vwap']) else float(completed['vwap'].iloc[-1])
    if live_vwap <= 0:
        return None

    trend_bullish = (
        current_price > live_vwap
        and current_price > live_ema_200
        and live_ema_200 >= prev_ema_200
    )
    if not trend_bullish:
        return None

    recent_completed = completed.tail(PULLBACK_LOOKBACK_BARS).copy()
    pullback_candidates = recent_completed[
        recent_completed['rsi_2'].notna() & (recent_completed['rsi_2'] <= PULLBACK_RSI_MAX)
    ]
    if pullback_candidates.empty:
        return None

    pullback_bar = pullback_candidates.iloc[-1]
    pullback_index = int(pullback_bar.name)
    prev_bar = completed.iloc[-1]

    big_green_count = _count_big_green_candles(completed.tail(PULLBACK_MAX_BIG_GREEN_CANDLES))
    if big_green_count >= PULLBACK_MAX_BIG_GREEN_CANDLES:
        return None

    live_rsi_input = pd.concat([completed['close'], pd.Series([current_price])], ignore_index=True)
    current_rsi_2 = calculate_rsi(live_rsi_input, period=PULLBACK_RSI_PERIOD)
    if current_rsi_2 is not None and current_rsi_2 > 70:
        return None

    vwap_stretch_pct = ((current_price - live_vwap) / live_vwap) * 100 if live_vwap else 0.0
    if vwap_stretch_pct > PULLBACK_VWAP_STRETCH_PERCENT:
        return None

    minor_resistance = float(completed.tail(PULLBACK_LOOKBACK_BARS)['high'].max())
    previous_candle_high = float(prev_bar['high'])
    trigger_buffer = 1 + (PULLBACK_ENTRY_BUFFER_PERCENT / 100)

    intraday_bar_avg = completed['session_volume'].tail(PULLBACK_INTRADAY_VOLUME_LOOKBACK).replace(0, np.nan).mean()
    signal_bar_volume = max(float(current_bar.get('session_volume', 0) or 0), float(prev_bar.get('session_volume', 0) or 0))
    intraday_volume_ratio = (
        signal_bar_volume / intraday_bar_avg
        if pd.notna(intraday_bar_avg) and intraday_bar_avg > 0 else 1.0
    )
    if intraday_volume_ratio < PULLBACK_INTRADAY_VOLUME_RATIO_MIN:
        return None

    entry_trigger = None
    entry_level = None
    if current_price >= minor_resistance * trigger_buffer:
        entry_trigger = 'MINOR_RESISTANCE_BREAK'
        entry_level = minor_resistance
    elif current_price >= previous_candle_high * trigger_buffer:
        entry_trigger = 'PREVIOUS_CANDLE_HIGH_BREAK'
        entry_level = previous_candle_high
    elif _is_bullish_engulfing(current_bar, prev_bar) and _is_strong_bullish_candle(current_bar):
        entry_trigger = 'BULLISH_ENGULFING'
        entry_level = previous_candle_high
    elif _is_strong_bullish_candle(current_bar) and current_price > previous_candle_high:
        entry_trigger = 'STRONG_BULLISH_CANDLE'
        entry_level = previous_candle_high
    else:
        return None

    stop_reference = min(float(pullback_bar['low']), live_vwap)
    stop_loss = round(stop_reference * (1 - PULLBACK_STOP_BUFFER_PERCENT / 100), 2)
    risk_per_share = current_price - stop_loss
    if risk_per_share <= 0:
        return None

    target_price = round(current_price + (risk_per_share * TARGET_PROFIT_MULTIPLIER), 2)
    trail_ema = float(current_bar['ema_trail']) if pd.notna(current_bar['ema_trail']) else None

    return {
        'symbol': symbol,
        'instrument_key': instrument_key,
        'entry_level': round(entry_level, 2),
        'entry_trigger': entry_trigger,
        'pullback_low': round(float(pullback_bar['low']), 2),
        'pullback_rsi_2': float(pullback_bar['rsi_2']),
        'pullback_time': pd.to_datetime(pullback_bar['date']).isoformat(),
        'vwap_value': round(live_vwap, 2),
        'ema_200': round(live_ema_200, 2),
        'current_rsi_2': round(float(current_rsi_2), 2) if current_rsi_2 is not None else None,
        'underlying_stop_loss': stop_loss,
        'underlying_target': target_price,
        'risk_per_share': round(risk_per_share, 2),
        'volume_ratio': round(float(intraday_volume_ratio), 2),
        'bars_since_pullback': len(completed) - 1 - pullback_index,
        'dynamic_trail_ema': round(trail_ema, 2) if trail_ema is not None else None,
    }

def init_one(access_token, args):
    key, symbol, yday = args
    try:
        key = norm_key(key)

        # Small throttle: 150ms between worker calls prevents 429 burst
        # when all 3 workers fire simultaneously across 205 stocks.
        time.sleep(0.15)

        # Use per-symbol lock to prevent duplicate concurrent fetches
        with get_cache_lock(symbol):
            vh = get_cached_or_fetch_candles(access_token, symbol, key)
        
        if vh is None or vh.empty:
            return None, 'no_data'
        
        weekday_data = vh[vh['date'].dt.weekday < 5]
        if len(weekday_data) < VOLUME_LOOKBACK_DAYS:
            return None, 'insufficient_data'
        
        avg_vol = weekday_data['volume'].tail(VOLUME_LOOKBACK_DAYS).mean()
        if not pd.notna(avg_vol) or avg_vol <= 0 or avg_vol < MIN_AVG_VOLUME:
            return None, 'volume_filtered'

        # ── FIX #2: Try to get yesterday OHLC from the already-loaded vh first.
        # This avoids a separate Upstox API call that will silently fail with an
        # expired/invalid token, causing every stock to be dropped as 'no_ohlc'.
        yday_str = yday.strftime('%Y-%m-%d') if hasattr(yday, 'strftime') else str(yday)
        yday_ts  = pd.Timestamp(yday_str)

        # Look for an exact date match inside the cached daily candles
        yday_row = vh[vh['date'].dt.normalize() == yday_ts]
        if not yday_row.empty:
            row = yday_row.iloc[-1]
            ohlc = {
                'date':   yday,
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row['volume']),
            }
            if DEBUG_MODE:
                print(f"📋 {symbol}: OHLC sourced from cache (no API call needed)")
        else:
            # Fallback: fetch from Upstox API (requires valid token)
            ohlc = fetch_historical_ohlc(access_token, key, yday)
            if not ohlc:
                # Last resort: use the most recent candle in the cache
                if not vh.empty:
                    row = vh.iloc[-1]
                    ohlc = {
                        'date':   row['date'],
                        'open':   float(row['open']),
                        'high':   float(row['high']),
                        'low':    float(row['low']),
                        'close':  float(row['close']),
                        'volume': float(row['volume']),
                    }
                    if DEBUG_MODE:
                        print(f"⚠️ {symbol}: OHLC API failed — using latest cache row ({row['date'].date()})")
                else:
                    return None, 'no_ohlc'
        
        pivot, r3 = calc_r3(ohlc['high'], ohlc['low'], ohlc['close'])
        _, s3 = calc_s3(ohlc['high'], ohlc['low'], ohlc['close'])
        
        # Reuse already-loaded vh for Klinger (avoids second fetch)
        klinger_data = None
        if ENABLE_KLINGER_FILTER:
            if len(vh) >= MIN_CANDLES_FOR_KLINGER:
                klinger, signal_line, histogram = calculate_klinger_adaptive(vh, symbol)
                if klinger is not None and len(klinger) >= 2:
                    ko_history_len = min(5, len(klinger))
                    ko_history_init = [float(klinger.iloc[-(ko_history_len - i)]) for i in range(ko_history_len - 1, -1, -1)]
                    klinger_data = {
                        'klinger': float(klinger.iloc[-1]),
                        'signal': float(signal_line.iloc[-1]),
                        'histogram': float(histogram.iloc[-1]),
                        'klinger_prev': float(klinger.iloc[-2]),
                        'signal_prev': float(signal_line.iloc[-2]),
                        'ko_history': ko_history_init,
                        'last_update': datetime.now(),
                        'candle_count': len(vh),
                        'adaptive_params': len(vh) < 90 if ADAPTIVE_KLINGER_LOOKBACK else False
                    }
                    if DEBUG_MODE:
                        adaptive_msg = " (Adaptive)" if klinger_data.get('adaptive_params') else ""
                        print(f" ✓ {symbol}: Klinger initialized (KO: {klinger_data['klinger']:.2f}{adaptive_msg})")
        
        return ({
            'key': key,
            'symbol': symbol,
            'r3': r3,
            's3': s3,
            'pivot': pivot,
            'yesterday_high': ohlc['high'],
            'yesterday_low': ohlc['low'],
            'yesterday_close': ohlc['close'],
            'avg_volume_20d': avg_vol,
            'box_high': ohlc['high'],
            'box_low': ohlc['low'],
            'klinger': klinger_data
        }, 'success')
    except Exception as e:
        if DEBUG_MODE:
            print(f"Init error for {symbol}: {e}")
        return None, 'error'

def reset_initialization():
    """Reset initialization state to allow retry"""
    global INITIALIZATION_RETRIES, R3_LEVELS, SYMBOL_TO_ISIN, ISIN_TO_SYMBOL, SYMBOL_TO_FO_KEY, VOLUME_DATA
    INITIALIZATION_RETRIES = 0
    R3_LEVELS.clear()
    SYMBOL_TO_ISIN.clear()
    ISIN_TO_SYMBOL.clear()
    SYMBOL_TO_FO_KEY.clear()
    VOLUME_DATA.clear()
    print("🔄 Initialization state reset")


def initialize_r3_levels(access_token, keys, symbols):
    global R3_LEVELS, SYMBOL_TO_ISIN, ISIN_TO_SYMBOL, SYMBOL_TO_FO_KEY, VOLUME_DATA, INITIALIZATION_RETRIES

    # Allow up to 3 retries
    if INITIALIZATION_RETRIES >= 3:
        print("❌ Max initialization retries reached")
        return False

    if INITIALIZATION_RETRIES > 0:
        print(f"\n🔄 Retry #{INITIALIZATION_RETRIES} of initialization...")
        time.sleep(5)  # Wait before retry

    INITIALIZATION_RETRIES += 1
    
    # Initialize cache directory
    if ENABLE_CANDLE_CACHE:
        init_cache_directory()
        load_cache_stats()
    
    ref = previous_trading_day()
    print(f"\n📊 Calculating R3/S3/Box levels + Klinger using {ref} data...")
    print(f"Using {MAX_WORKERS} workers | Min Volume: {MIN_AVG_VOLUME:,}")
    if ENABLE_KLINGER_FILTER:
        print(f"🔥 Klinger Oscillator: ENABLED (Fast={KLINGER_FAST}, Slow={KLINGER_SLOW}, Signal={KLINGER_SIGNAL})")
        if ADAPTIVE_KLINGER_LOOKBACK:
            print(f"   Adaptive Mode: ON (Short: Fast={KLINGER_FAST_SHORT}, Slow={KLINGER_SLOW_SHORT}, Signal={KLINGER_SIGNAL_SHORT})")
    if ENABLE_CANDLE_CACHE:
        print(f"💾 Candle Cache: ENABLED (Min candles: {MIN_CANDLES_FOR_KLINGER})")
    print()
    
    ok = volf = no_data = insufficient = no_ohlc = 0
    tasks = [(k, symbols.get(k) or symbols.get(norm_key(k)) or k.split('|')[-1].split(':')[-1], ref) for k in keys]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(init_one, access_token, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            if i % 25 == 0:
                print(f" Progress: {i}/{len(keys)} | Success: {ok}")
            res, status = f.result()
            if status == 'success' and res:
                nk = norm_key(res['key'])
                R3_LEVELS[nk] = {
                    'symbol': res['symbol'],
                    'r3': res['r3'],
                    's3': res['s3'],
                    'pivot': res['pivot'],
                    'yesterday_high': res['yesterday_high'],
                    'yesterday_low': res['yesterday_low'],
                    'yesterday_close': res['yesterday_close'],
                    'avg_volume_20d': res['avg_volume_20d'],
                    'box_high': res.get('box_high', res['yesterday_high']),
                    'box_low': res.get('box_low', res['yesterday_low']),
                    'klinger': res.get('klinger')
                }
                VOLUME_DATA[nk] = res['avg_volume_20d']
                ok += 1
            elif status == 'volume_filtered':
                volf += 1
            elif status == 'no_data':
                no_data += 1
            elif status == 'insufficient_data':
                insufficient += 1
            elif status == 'no_ohlc':
                no_ohlc += 1
    
    SYMBOL_TO_ISIN = {info['symbol']: isin_key for isin_key, info in R3_LEVELS.items()}
    ISIN_TO_SYMBOL = {isin_key: info['symbol'] for isin_key, info in R3_LEVELS.items()}

    # Build SYMBOL_TO_FO_KEY: map each symbol -> its NSE_FO continuous/spot instrument_key.
    # This is used as a fallback in fetch_5min_candle_data when Upstox intraday/historical
    # endpoints are tried (they require an NSE_FO key rather than an NSE_EQ key).
    try:
        _inst_url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
        _inst_df = pd.read_csv(_inst_url, compression='gzip')
        # Keep only NSE_FO rows; prefer EQ/XX (spot/index futures) instrument types
        _fo_df = _inst_df[_inst_df['exchange'] == 'NSE_FO'].copy()
        # Strip expiry suffix to get the base symbol (e.g. "RELIANCE24JUL25000CE" -> "RELIANCE")
        _fo_df['base_sym'] = (
            _fo_df['tradingsymbol']
            .str.replace(r'\d{2}[A-Z]{3}\d{2,4}.*', '', regex=True)
            .str.strip()
        )
        # For each base symbol keep the row whose instrument_key we prefer.
        # Priority: instrument_type == 'EQ' (equity futures spot) first, then first available.
        _fo_eq = _fo_df[_fo_df.get('instrument_type', pd.Series(dtype=str)) == 'EQ']
        if 'instrument_type' in _fo_df.columns:
            _fo_eq = _fo_df[_fo_df['instrument_type'] == 'EQ']
            _fo_other = _fo_df[~(_fo_df['instrument_type'] == 'EQ')]
        else:
            _fo_eq = _fo_df.iloc[0:0]  # empty
            _fo_other = _fo_df
        # Build mapping: first from EQ-type rows, then fill gaps from remaining rows
        _fo_key_map = {}
        for _df_part in [_fo_other, _fo_eq]:  # lower priority first so EQ wins
            for _, _row in _df_part.iterrows():
                _fo_key_map[_row['base_sym']] = _row['instrument_key']
        # Only populate for symbols we actually track
        SYMBOL_TO_FO_KEY = {sym: _fo_key_map[sym] for sym in SYMBOL_TO_ISIN if sym in _fo_key_map}
        print(f"✅ SYMBOL_TO_FO_KEY built: {len(SYMBOL_TO_FO_KEY)} symbols mapped to NSE_FO keys")
    except Exception as _e:
        print(f"⚠️ Could not build SYMBOL_TO_FO_KEY ({_e}); Upstox 5min fallback will use NSE_EQ keys")
    
    # Count Klinger success
    klinger_success = sum(1 for info in R3_LEVELS.values() if info.get('klinger') is not None)
    klinger_adaptive = sum(1 for info in R3_LEVELS.values() 
                          if info.get('klinger') and info['klinger'].get('adaptive_params'))
    
    print(f"\n{'='*100}")
    print(f"✅ Successfully initialized: {ok} stocks")
    if ENABLE_KLINGER_FILTER:
        print(f"🔥 Klinger data available: {klinger_success} stocks ({(klinger_success/ok*100):.1f}%)")
        if ADAPTIVE_KLINGER_LOOKBACK and klinger_adaptive > 0:
            print(f"   Adaptive parameters used: {klinger_adaptive} stocks")
    if ENABLE_CANDLE_CACHE:
        print(f"💾 Cache Statistics:")
        print(f"   Cache Hits: {CACHE_STATS['cache_hits']}")
        print(f"   Cache Misses: {CACHE_STATS['cache_misses']}")
        if CACHE_STATS['cache_hits'] + CACHE_STATS['cache_misses'] > 0:
            hit_rate = CACHE_STATS['cache_hits'] / (CACHE_STATS['cache_hits'] + CACHE_STATS['cache_misses']) * 100
            print(f"   Hit Rate: {hit_rate:.1f}%")
    total_filtered = volf + no_data + insufficient + no_ohlc
    print(f"⚠️ Filtered out: {total_filtered} stocks")
    if total_filtered > 0:
        print(f"   • Volume too low  : {volf}")
        print(f"   • No candle data  : {no_data}")
        print(f"   • Insufficient days: {insufficient}")
        print(f"   • No OHLC data    : {no_ohlc}")
    print(f"{'='*100}\n")
    
    # Save cache stats
    if ENABLE_CANDLE_CACHE:
        save_cache_stats()

    if ok == 0 and INITIALIZATION_RETRIES < 3:
        print("⚠️ No stocks initialized, will retry...")
        return False

    return ok > 0

# ========== R3/S3 BREAKOUT DETECTION ==========

# Staleness config: if first confirmation has not been followed by a second
# within MAX_SCANS_WITHOUT_PROGRESS scans, reset the counter entirely.
# This fixes AMBUJACEM-type scenarios where count never reaches 2/2.
MAX_SCANS_WITHOUT_PROGRESS = 3   # ~90 seconds at 30s per scan

def reset_stale_breach_states():
    """Clean up old breach states that weren't confirmed.
    Also resets counters that have been stuck at count=1 for too many scans
    (AMBUJACEM-type fix: confirmation counter resets after 3 scans with no progress).
    """
    current_time = datetime.now()
    stale_keys = []
    for key, state in LAST_BREAKOUT_STATE.items():
        elapsed = (current_time - state['first_breach_time']).seconds
        # Standard: expired window
        if elapsed > BREACH_TIME_WINDOW:
            stale_keys.append(key)
            continue
        # New: stuck at count=1 with no second confirmation in 3 scans
        scans_since_last = state.get('scans_since_last_breach', 0) + 1
        state['scans_since_last_breach'] = scans_since_last
        if state['breach_count'] == 1 and scans_since_last >= MAX_SCANS_WITHOUT_PROGRESS:
            stale_keys.append(key)
            if DEBUG_MODE:
                symbol = ISIN_TO_SYMBOL.get(key, key)
                print(f"🔄 {symbol}: Confirmation counter reset (stuck at 1/{BREACH_CONFIRMATION_CYCLES} "
                      f"for {scans_since_last} scans)")
    for key in stale_keys:
        del LAST_BREAKOUT_STATE[key]

def check_breakout_legacy(key, live):
    """Enhanced R3 breakout check with false alert prevention"""
    info = R3_LEVELS.get(key)
    if not info:
        return None

    # ── SECOND-HALF REVERSE WATCH (R3 LONG) ─────────────────────────────────
    # Already fired R3 LONG → skip entirely.
    # Fired S3 SHORT earlier → allow R3 LONG re-entry only in 2nd half (reversal bounce).
    if info['symbol'] in R3_ALERTED_STOCKS:
        return None   # already took the R3 LONG today
    _in_second_half_r3 = (ENABLE_SECOND_HALF_SHORT_REWATCH
                          and datetime.now().strftime("%H:%M") >= SECOND_HALF_START)
    if info['symbol'] in S3_ALERTED_STOCKS and not _in_second_half_r3:
        return None   # had S3 SHORT but too early for reverse LONG
    
    if live['high'] is None or info['r3'] is None or info['r3'] == 0:
        return None
    current_price = live['ltp']
    r3_level = info['r3']
    
    # VALIDATION 1: Basic Breach Check
    if live['high'] < r3_level:
        if key in LAST_BREAKOUT_STATE:
            del LAST_BREAKOUT_STATE[key]
        return None
    
    # VALIDATION 2: Price Sustainability
    # When ltp dips below threshold temporarily, preserve state — don't delete on noise.
    # Let BREACH_TIME_WINDOW handle expiry.
    sustainability_threshold = r3_level * (1 + PRICE_SUSTAINABILITY_PERCENT / 100)
    if current_price < sustainability_threshold:
        if DEBUG_MODE:
            print(f"⚠️ {info['symbol']}: Touched R3 but price not sustainable " 
                  f"(Current: ₹{current_price:.2f} vs Required: ₹{sustainability_threshold:.2f})")
        if key in LAST_BREAKOUT_STATE:
            LAST_BREAKOUT_STATE[key]['scans_since_last_breach'] = \
                LAST_BREAKOUT_STATE[key].get('scans_since_last_breach', 0) + 1
        return None
    
    # VALIDATION 3: Volume Check
    cur_vol = live.get('volume') or 0
    avg20 = info['avg_volume_20d']
    if avg20 <= 0:
        return None
    ratio = cur_vol / avg20
    thr = dynamic_volume_threshold()

    # ── FII/DII TREND RELIEF (LONG/CE) ───────────────────────────────────────
    # Volume threshold relaxation based on multi-day institutional stance:
    #   score >= 2 (strong accumulation / unusual reversal) → 10% relief
    #   score == 1 (FII leading buy / DII sell)             →  5% relief
    #   score <= 0                                          → label only, no relief
    fii_trend_score = 0
    fii_trend_label = ""
    if ENABLE_FII_DII_TREND_FILTER:
        fii_trend_score = get_fii_dii_trend_score(info['symbol'])
        if fii_trend_score >= 2:
            thr = thr * FII_DII_TREND_VOLUME_RELIEF          # 10% relief
            fii_trend_label = f" [FII score={fii_trend_score:+d} → vol thr relaxed to {thr:.2f}x]"
        elif fii_trend_score == 1:
            thr = thr * (1 - (1 - FII_DII_TREND_VOLUME_RELIEF) / 2)  # 5% relief
            fii_trend_label = f" [FII score={fii_trend_score:+d} → vol thr relaxed to {thr:.2f}x]"
        elif fii_trend_score < 0:
            fii_trend_label = f" [FII score={fii_trend_score:+d} headwind]"
    # ── END FII/DII TREND RELIEF ──────────────────────────────────────────────

    if ratio < thr:
        return None

    # VALIDATION 4: Consecutive Confirmation
    current_time = datetime.now()
    if key not in LAST_BREAKOUT_STATE:
        LAST_BREAKOUT_STATE[key] = {
            'breach_count': 1,
            'first_breach_time': current_time,
            'last_breach_time': current_time,
            'breach_type': 'R3',
            'max_price': current_price,
            'volume_ratios': [ratio],
            'scans_since_last_breach': 0
        }
        if DEBUG_MODE:
            print(f"📊 {info['symbol']}: First R3 breach detected at ₹{current_price:.2f} "
                  f"(1/{BREACH_CONFIRMATION_CYCLES} confirmations){fii_trend_label}")
        return None
    
    state = LAST_BREAKOUT_STATE[key]
    time_since_first = (current_time - state['first_breach_time']).seconds
    if time_since_first > BREACH_TIME_WINDOW:
        LAST_BREAKOUT_STATE[key] = {
            'breach_count': 1,
            'first_breach_time': current_time,
            'last_breach_time': current_time,
            'breach_type': 'R3',
            'max_price': current_price,
            'volume_ratios': [ratio],
            'scans_since_last_breach': 0
        }
        if DEBUG_MODE:
            print(f"⏳ {info['symbol']}: Breach window expired, restarting confirmation")
        return None
    
    state['breach_count'] += 1
    state['last_breach_time'] = current_time
    state['max_price'] = max(state['max_price'], current_price)
    state['volume_ratios'].append(ratio)
    state['scans_since_last_breach'] = 0  # reset staleness counter on progress
    
    if state['breach_count'] >= BREACH_CONFIRMATION_CYCLES:
        # VALIDATION 5: Volume Persistence
        avg_volume_ratio = sum(state['volume_ratios']) / len(state['volume_ratios'])
        if avg_volume_ratio < thr * 0.9:
            if DEBUG_MODE:
                print(f"⚠️ {info['symbol']}: Volume not persistent " 
                      f"(Avg: {avg_volume_ratio:.2f}x vs Required: {thr*0.9:.2f}x)")
            del LAST_BREAKOUT_STATE[key]
            return None
        
        # VALIDATION 6: Price Momentum
        price_gain = ((current_price - r3_level) / r3_level) * 100
        if price_gain < 0.3:
            if DEBUG_MODE:
                print(f"⚠️ {info['symbol']}: Insufficient momentum above R3 (Gain: {price_gain:.2f}%)")
            return None
        
        # ✅ ALL VALIDATIONS PASSED
        print(f"\n✅ {info['symbol']}: R3 BREAKOUT CONFIRMED!")
        print(f" Confirmations: {state['breach_count']} | Time: {time_since_first}s")
        print(f" Max price: ₹{state['max_price']:.2f} | Avg volume: {avg_volume_ratio:.2f}x")
        del LAST_BREAKOUT_STATE[key]
        
        return {
            'symbol': info['symbol'],
            'instrument_key': key,
            'r3': r3_level,
            'current_price': current_price,
            'high': live['high'],
            'volume_ratio': ratio,
            'current_volume': cur_vol,
            'avg_volume': avg20,
            'timestamp': current_time,
            'yesterday_close': info['yesterday_close'],
            'volume_threshold_used': thr,
            'breakout_type': 'CE',
            'strategy': 'R3',
            'confirmation_cycles': state['breach_count'],
            'time_to_confirm': time_since_first
        }
    
    if DEBUG_MODE:
        print(f"📊 {info['symbol']}: R3 breach #{state['breach_count']}/{BREACH_CONFIRMATION_CYCLES}")
    return None

def check_breakout(key, live):
    """Refined CE pullback-breakout check with confirmation and exhaustion filters."""
    info = R3_LEVELS.get(key)
    if not info:
        return None

    if info['symbol'] in R3_ALERTED_STOCKS:
        return None

    in_second_half_r3 = (
        ENABLE_SECOND_HALF_SHORT_REWATCH
        and datetime.now().strftime("%H:%M") >= SECOND_HALF_START
    )
    if info['symbol'] in S3_ALERTED_STOCKS and not in_second_half_r3:
        return None

    current_price = live.get('ltp')
    if not current_price or not _UPSTOX_SESSION_TOKEN:
        return None

    setup = build_pullback_ce_signal(_UPSTOX_SESSION_TOKEN, key, info, live)
    if not setup:
        if key in LAST_BREAKOUT_STATE and LAST_BREAKOUT_STATE[key].get('breach_type') == 'PULLBACK_CE':
            del LAST_BREAKOUT_STATE[key]
        return None

    current_time = datetime.now()
    setup_anchor = setup['pullback_time']
    state = LAST_BREAKOUT_STATE.get(key)

    if (
        state is None
        or state.get('breach_type') != 'PULLBACK_CE'
        or state.get('setup_anchor') != setup_anchor
    ):
        LAST_BREAKOUT_STATE[key] = {
            'breach_count': 1,
            'first_breach_time': current_time,
            'last_breach_time': current_time,
            'breach_type': 'PULLBACK_CE',
            'setup_anchor': setup_anchor,
            'max_price': current_price,
            'volume_ratios': [setup['volume_ratio']],
            'scans_since_last_breach': 0,
        }
        if DEBUG_MODE:
            print(
                f"📊 {info['symbol']}: pullback CE setup detected at ₹{current_price:.2f} "
                f"(trigger={setup['entry_trigger']}, 1/{BREACH_CONFIRMATION_CYCLES} confirmations)"
            )
        return None

    time_since_first = (current_time - state['first_breach_time']).seconds
    if time_since_first > BREACH_TIME_WINDOW:
        LAST_BREAKOUT_STATE[key] = {
            'breach_count': 1,
            'first_breach_time': current_time,
            'last_breach_time': current_time,
            'breach_type': 'PULLBACK_CE',
            'setup_anchor': setup_anchor,
            'max_price': current_price,
            'volume_ratios': [setup['volume_ratio']],
            'scans_since_last_breach': 0,
        }
        if DEBUG_MODE:
            print(f"⏳ {info['symbol']}: Pullback setup refreshed, restarting confirmation")
        return None

    state['breach_count'] += 1
    state['last_breach_time'] = current_time
    state['max_price'] = max(state['max_price'], current_price)
    state['volume_ratios'].append(setup['volume_ratio'])
    state['scans_since_last_breach'] = 0

    if state['breach_count'] >= BREACH_CONFIRMATION_CYCLES:
        avg_volume_ratio = sum(state['volume_ratios']) / len(state['volume_ratios'])
        if avg_volume_ratio < PULLBACK_INTRADAY_VOLUME_RATIO_MIN:
            if DEBUG_MODE:
                print(
                    f"⚠️ {info['symbol']}: intraday volume faded "
                    f"(avg {avg_volume_ratio:.2f}x < {PULLBACK_INTRADAY_VOLUME_RATIO_MIN:.2f}x)"
                )
            del LAST_BREAKOUT_STATE[key]
            return None

        print(f"\n✅ {info['symbol']}: PULLBACK CE SETUP CONFIRMED!")
        print(f" Confirmations: {state['breach_count']} | Time: {time_since_first}s")
        print(
            f" Trigger: {setup['entry_trigger']} | Pullback RSI(2): {setup['pullback_rsi_2']:.1f} "
            f"| Avg intraday volume: {avg_volume_ratio:.2f}x"
        )
        del LAST_BREAKOUT_STATE[key]

        return {
            'symbol': info['symbol'],
            'instrument_key': key,
            'level': setup['entry_level'],
            'current_price': current_price,
            'high': live.get('high'),
            'volume_ratio': setup['volume_ratio'],
            'current_volume': live.get('volume') or 0,
            'avg_volume': info.get('avg_volume_20d', 0),
            'timestamp': current_time,
            'yesterday_close': info['yesterday_close'],
            'breakout_type': 'CE',
            'strategy': 'PULLBACK_CE',
            'confirmation_cycles': state['breach_count'],
            'time_to_confirm': time_since_first,
            'entry_trigger': setup['entry_trigger'],
            'pullback_low': setup['pullback_low'],
            'pullback_rsi_2': setup['pullback_rsi_2'],
            'current_rsi_2': setup['current_rsi_2'],
            'vwap_value': setup['vwap_value'],
            'ema_200': setup['ema_200'],
            'underlying_stop_loss': setup['underlying_stop_loss'],
            'underlying_target': setup['underlying_target'],
            'risk_per_share': setup['risk_per_share'],
            'trail_activation_price': setup['underlying_target'],
            'dynamic_trail_ema': setup['dynamic_trail_ema'],
            'setup_anchor': setup_anchor,
        }

    if DEBUG_MODE:
        print(
            f"📊 {info['symbol']}: pullback CE confirmation "
            f"#{state['breach_count']}/{BREACH_CONFIRMATION_CYCLES}"
        )
    return None

def check_breakdown(key, live):
    """Enhanced S3 breakdown check with false alert prevention"""
    info = R3_LEVELS.get(key)
    if not info:
        return None

    # ── SECOND-HALF REVERSE WATCH (S3) ───────────────────────────────────────
    # If already fired S3 SHORT, skip entirely.
    # If fired R3 LONG earlier and now in 2nd half → allow S3 SHORT re-entry.
    if info['symbol'] in S3_ALERTED_STOCKS:
        return None   # already took the S3 SHORT today
    _in_second_half_s3 = (ENABLE_SECOND_HALF_SHORT_REWATCH
                          and datetime.now().strftime("%H:%M") >= SECOND_HALF_START)
    if info['symbol'] in R3_ALERTED_STOCKS and not _in_second_half_s3:
        return None   # had R3 LONG but too early for reverse SHORT
    
    if live['low'] is None or info.get('s3') is None or info['s3'] == 0:
        return None
    
    current_price = live['ltp']
    s3_level = info['s3']
    
    # VALIDATION 1: Basic Breach Check
    if live['low'] > s3_level:
        if key in LAST_BREAKOUT_STATE and LAST_BREAKOUT_STATE[key].get('breach_type') == 'S3':
            del LAST_BREAKOUT_STATE[key]
        return None
    
    # VALIDATION 2: Price Sustainability
    # KEY FIX: When ltp bounces above threshold temporarily, preserve state — don't
    # delete on noise. Bump staleness counter, let BREACH_TIME_WINDOW handle expiry.
    sustainability_threshold = s3_level * (1 - PRICE_SUSTAINABILITY_PERCENT / 100)
    if current_price > sustainability_threshold:
        if DEBUG_MODE:
            print(f"⚠️ {info['symbol']}: Touched S3 but price not sustainable " 
                  f"(Current: ₹{current_price:.2f} vs Required: ₹{sustainability_threshold:.2f})")
        if key in LAST_BREAKOUT_STATE and LAST_BREAKOUT_STATE[key].get('breach_type') == 'S3':
            LAST_BREAKOUT_STATE[key]['scans_since_last_breach'] = \
                LAST_BREAKOUT_STATE[key].get('scans_since_last_breach', 0) + 1
        return None
    
    # VALIDATION 3: Volume Check
    cur_vol = live.get('volume') or 0
    avg20 = info['avg_volume_20d']
    if avg20 <= 0:
        return None
    ratio = cur_vol / avg20
    thr = dynamic_volume_threshold()

    # ── FII/DII TREND RELIEF (SHORT/PE) ──────────────────────────────────────
    # Volume threshold relaxation for institutional distribution signals:
    #   score >= 2  → suppress PE entirely (FII accumulating = contradicts short)
    #   score == -1 → 5% relief  (FII distributing — moderate conviction)
    #   score <= -2 → 10% relief (strong distribution / unusual sell reversal)
    fii_trend_score_bd = 0
    fii_trend_label_bd = ""
    if ENABLE_FII_DII_TREND_FILTER:
        fii_trend_score_bd = get_fii_dii_trend_score(info['symbol'])
        if fii_trend_score_bd >= 2:
            # Strong institutional buying — suppress PE breakdown signal
            if DEBUG_MODE:
                print(f"⛔ {info['symbol']}: S3 breakdown suppressed — FII trend score "
                      f"{fii_trend_score_bd:+d} (strong accumulation contradicts PE)")
            return None
        elif fii_trend_score_bd <= -2:
            thr = thr * FII_DII_TREND_VOLUME_RELIEF           # 10% relief
            fii_trend_label_bd = f" [FII score={fii_trend_score_bd:+d} → vol thr relaxed to {thr:.2f}x]"
        elif fii_trend_score_bd == -1:
            thr = thr * (1 - (1 - FII_DII_TREND_VOLUME_RELIEF) / 2)  # 5% relief
            fii_trend_label_bd = f" [FII score={fii_trend_score_bd:+d} → vol thr relaxed to {thr:.2f}x]"
        elif fii_trend_score_bd > 0:
            fii_trend_label_bd = f" [FII score={fii_trend_score_bd:+d} headwind for PE]"
    # ── END FII/DII TREND RELIEF ──────────────────────────────────────────────

    if ratio < thr:
        return None

    # VALIDATION 4: Consecutive Confirmation
    current_time = datetime.now()
    if key not in LAST_BREAKOUT_STATE or LAST_BREAKOUT_STATE[key].get('breach_type') != 'S3':
        LAST_BREAKOUT_STATE[key] = {
            'breach_count': 1,
            'first_breach_time': current_time,
            'last_breach_time': current_time,
            'breach_type': 'S3',
            'min_price': current_price,
            'volume_ratios': [ratio]
        }
        if DEBUG_MODE:
            print(f"📊 {info['symbol']}: First S3 breach detected at ₹{current_price:.2f} "
                  f"(1/{BREACH_CONFIRMATION_CYCLES} confirmations){fii_trend_label_bd}")
        return None
    
    state = LAST_BREAKOUT_STATE[key]
    time_since_first = (current_time - state['first_breach_time']).seconds
    if time_since_first > BREACH_TIME_WINDOW:
        LAST_BREAKOUT_STATE[key] = {
            'breach_count': 1,
            'first_breach_time': current_time,
            'last_breach_time': current_time,
            'breach_type': 'S3',
            'min_price': current_price,
            'volume_ratios': [ratio]
        }
        if DEBUG_MODE:
            print(f"⏳ {info['symbol']}: Breach window expired, restarting confirmation")
        return None
    
    state['breach_count'] += 1
    state['last_breach_time'] = current_time
    state['min_price'] = min(state['min_price'], current_price)
    state['volume_ratios'].append(ratio)
    
    if state['breach_count'] >= BREACH_CONFIRMATION_CYCLES:
        # VALIDATION 5: Volume Persistence
        avg_volume_ratio = sum(state['volume_ratios']) / len(state['volume_ratios'])
        if avg_volume_ratio < thr * 0.9:
            if DEBUG_MODE:
                print(f"⚠️ {info['symbol']}: Volume not persistent " 
                      f"(Avg: {avg_volume_ratio:.2f}x vs Required: {thr*0.9:.2f}x)")
            del LAST_BREAKOUT_STATE[key]
            return None
        
        # VALIDATION 6: Price Momentum
        price_drop = ((s3_level - current_price) / s3_level) * 100
        if price_drop < 0.3:
            if DEBUG_MODE:
                print(f"⚠️ {info['symbol']}: Insufficient momentum below S3 (Drop: {price_drop:.2f}%)")
            return None
        
        # ✅ ALL VALIDATIONS PASSED
        print(f"\n✅ {info['symbol']}: S3 BREAKDOWN CONFIRMED!")
        print(f" Confirmations: {state['breach_count']} | Time: {time_since_first}s")
        print(f" Min price: ₹{state['min_price']:.2f} | Avg volume: {avg_volume_ratio:.2f}x")
        del LAST_BREAKOUT_STATE[key]
        
        return {
            'symbol': info['symbol'],
            'instrument_key': key,
            's3': s3_level,
            'current_price': current_price,
            'low': live['low'],
            'volume_ratio': ratio,
            'current_volume': cur_vol,
            'avg_volume': avg20,
            'timestamp': current_time,
            'yesterday_close': info['yesterday_close'],
            'volume_threshold_used': thr,
            'breakout_type': 'PE',
            'strategy': 'S3',
            'confirmation_cycles': state['breach_count'],
            'time_to_confirm': time_since_first
        }
    
    if DEBUG_MODE:
        print(f"📊 {info['symbol']}: S3 breach #{state['breach_count']}/{BREACH_CONFIRMATION_CYCLES}")
    return None

# ========== BOX THEORY FUNCTIONS WITH KLINGER ==========
def check_exit_conditions(position, current_price, trader):
    """Check if any exit condition is met for a position"""
    global DAILY_PNL, POSITION_PEAK_PRICES, POSITION_TRAILING_SL
    
    if not ENABLE_EXIT_MANAGEMENT:
        return False, None
    
    position_id = position.get('order_id') or position.get('position_id')
    symbol = position['symbol']
    entry_price = position['entry_price']
    strategy = position.get('strategy', 'UNKNOWN')
    underlying_key = position.get('underlying_key')
    
    # Calculate current P&L
    pnl_per_unit = current_price - entry_price
    total_pnl = pnl_per_unit * position['quantity']
    pnl_percent = (pnl_per_unit / entry_price) * 100
    
    # Update position with current P&L
    position['current_price'] = current_price
    position['current_pnl'] = total_pnl
    position['pnl_percent'] = pnl_percent
    
    # 1. TIME-BASED EXIT (Most Important - Before Market Close)
    if ENABLE_TIME_BASED_EXIT:
        now = datetime.now()
        current_time_str = now.strftime("%H:%M")
        
        # Regular end-of-day exit
        if current_time_str >= EXIT_START_TIME:
            return True, "END_OF_DAY"
        
        # 2. EXPIRY DAY EXIT (Exit earlier on expiry day)
        if ENABLE_EXPIRY_DAY_EXIT:
            expiry_date = position.get('expiry_date')
            if expiry_date:
                if isinstance(expiry_date, str):
                    expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")
                
                if expiry_date.date() == now.date():
                    if current_time_str >= EXPIRY_EXIT_TIME:
                        return True, "EXPIRY_DAY_EXIT"
    
    # 3. MAXIMUM DAILY LOSS CHECK
    if DAILY_PNL + total_pnl <= -MAX_DAILY_LOSS:
        return True, "MAX_DAILY_LOSS"
    
    # 4. MAXIMUM DAILY PROFIT CHECK
    if DAILY_PNL + total_pnl >= MAX_DAILY_PROFIT:
        return True, "MAX_DAILY_PROFIT"

    # 5. STRUCTURAL EXIT FLOW FOR PULLBACK CE
    if strategy == 'PULLBACK_CE' and underlying_key:
        underlying_price = trader.get_ltp(underlying_key)
        if underlying_price:
            position['underlying_ltp'] = underlying_price
            structural_stop = position.get('underlying_stop_loss')
            rr_target = position.get('underlying_target')

            if structural_stop and underlying_price <= structural_stop:
                return True, f"STRUCTURAL_STOP (Underlying: ₹{underlying_price:.2f} <= ₹{structural_stop:.2f})"

            if rr_target and underlying_price >= rr_target:
                position['rr_target_hit'] = True

            if position.get('rr_target_hit'):
                access_token = _UPSTOX_SESSION_TOKEN
                df5 = fetch_5min_cached(access_token, underlying_key, bars=80, symbol=symbol) if access_token else None
                if df5 is not None and len(df5) >= 10:
                    df_ind = build_intraday_indicator_frame(df5)
                    if df_ind is not None:
                        today_df = df_ind[df_ind['session_date'] == datetime.now().date()].copy().reset_index(drop=True)
                        if len(today_df) >= 2:
                            prev_completed = today_df.iloc[-2]
                            latest_bar = today_df.iloc[-1]
                            trail_candidates = [float(prev_completed['low'])]
                            if pd.notna(latest_bar['ema_trail']):
                                trail_candidates.append(float(latest_bar['ema_trail']))
                            dynamic_trail = round(max(trail_candidates), 2)
                            existing_trail = position.get('dynamic_trail_level')
                            if existing_trail is not None:
                                dynamic_trail = max(dynamic_trail, round(float(existing_trail), 2))
                            position['dynamic_trail_level'] = dynamic_trail
                            if underlying_price <= dynamic_trail:
                                return True, f"EMA_PREV_LOW_TRAIL (Underlying: ₹{underlying_price:.2f} <= ₹{dynamic_trail:.2f})"
                            return False, None
                return True, f"RR_TARGET_2R (Underlying: ₹{underlying_price:.2f} >= ₹{rr_target:.2f})"
        return False, None
    
    # 6. TARGET PROFIT REACHED (2x risk or configured multiplier)
    risk_amount = entry_price * (STOPLOSS_PERCENTAGE / 100)
    target_profit = risk_amount * TARGET_PROFIT_MULTIPLIER
    if pnl_per_unit >= target_profit:
        return True, "TARGET_PROFIT"
    
    # 7. TRAILING STOP-LOSS (Activate after specified profit %)
    #    Stage 1 – initial SL order is placed immediately after entry (see entry logic).
    #    Stage 2 – here we monitor price and MODIFY the live SL order on the broker
    #              every time price reaches a new peak, ratcheting the stop upward.
    if ENABLE_TRAILING_STOP and pnl_percent >= TRAILING_STOP_ACTIVATION:
        # --- update in-memory peak ---
        if position_id not in POSITION_PEAK_PRICES:
            POSITION_PEAK_PRICES[position_id] = current_price
        else:
            POSITION_PEAK_PRICES[position_id] = max(POSITION_PEAK_PRICES[position_id], current_price)

        peak_price            = POSITION_PEAK_PRICES[position_id]
        new_trailing_sl       = round(peak_price * (1 - TRAILING_STOP_PERCENTAGE / 100), 2)
        last_broker_sl        = POSITION_TRAILING_SL.get(position_id)
        sl_order_id           = position.get('sl_order_id')

        # --- push updated trigger to broker only when SL has moved up meaningfully ---
        MIN_SL_MOVE_PCT = 0.10   # only modify if SL shifted by ≥ 0.10 % (avoids API noise)
        should_modify = (
            sl_order_id is not None
            and (
                last_broker_sl is None                                        # first time activating
                or new_trailing_sl > last_broker_sl * (1 + MIN_SL_MOVE_PCT / 100)  # moved up
            )
        )
        if should_modify:
            modify_result = trader.modify_order(
                order_id      = sl_order_id,
                trigger_price = new_trailing_sl,
                price         = 0,                          # SL-M: execute at market when triggered
                quantity      = position.get('quantity'),
                order_type    = 'SL',                       # SL-M
            )
            if modify_result.get('status_code') == 200:
                POSITION_TRAILING_SL[position_id] = new_trailing_sl
                print(f"   🔒 Trailing SL updated → ₹{new_trailing_sl:.2f}  (peak ₹{peak_price:.2f})")
            else:
                # Broker modify failed – fall back to software-side check so we still exit
                print(f"   ⚠️ Broker SL modify failed; software-side guard still active.")

        # --- software-side safety net (catches the stop even if modify failed) ---
        effective_sl = POSITION_TRAILING_SL.get(position_id, new_trailing_sl)
        if current_price <= effective_sl:
            return True, f"TRAILING_STOP (Peak: ₹{peak_price:.2f}, SL: ₹{effective_sl:.2f})"
    
    # 8. STRATEGY-SPECIFIC EXITS
    if ENABLE_STRATEGY_EXITS and underlying_key:
        # BOX THEORY: Exit if price re-enters the box
        if strategy in ['BOX_TOP', 'BOX_BOTTOM']:
            underlying_price = trader.get_ltp(underlying_key)
            info = R3_LEVELS.get(underlying_key)
            
            if info and underlying_price:
                box_high = info['box_high']
                box_low = info['box_low']
                
                if strategy == 'BOX_TOP':
                    # Exit if underlying price drops back into box
                    reentry_threshold = box_high * (1 - BOX_REENTRY_EXIT_PERCENT / 100)
                    if underlying_price < reentry_threshold:
                        return True, f"BOX_REENTRY (Price: ₹{underlying_price:.2f} < ₹{reentry_threshold:.2f})"
                
                elif strategy == 'BOX_BOTTOM':
                    # Exit if underlying price rises back into box
                    reentry_threshold = box_low * (1 + BOX_REENTRY_EXIT_PERCENT / 100)
                    if underlying_price > reentry_threshold:
                        return True, f"BOX_REENTRY (Price: ₹{underlying_price:.2f} > ₹{reentry_threshold:.2f})"
        
        # RANGE TRADING: Exit if support/resistance breaks
        elif strategy in ['BOUNCE_BOTTOM', 'REJECT_TOP']:
            underlying_price = trader.get_ltp(underlying_key)
            info = R3_LEVELS.get(underlying_key)
            
            if info and underlying_price:
                if strategy == 'BOUNCE_BOTTOM':
                    # Exit if support breaks
                    if underlying_price < info['box_low'] * 0.995:
                        return True, f"SUPPORT_BROKEN (Price: ₹{underlying_price:.2f})"
                
                elif strategy == 'REJECT_TOP':
                    # Exit if resistance breaks
                    if underlying_price > info['box_high'] * 1.005:
                        return True, f"RESISTANCE_BROKEN (Price: ₹{underlying_price:.2f})"
        
        # GAP TRADING: Exit when gap fills significantly
        elif position.get('trade_type') == 'GAP_OPTION':
            gap_signal = position.get('gap_signal')
            if gap_signal == "gap_fill":
                # Check if gap has filled
                gap_info = GAP_LEVELS.get(symbol)
                if gap_info:
                    underlying_price = trader.get_ltp(underlying_key)
                    if underlying_price:
                        gap_info['current_price'] = underlying_price
                        fill_percent = calculate_gap_fill_percent(gap_info)
                        
                        if fill_percent >= GAP_FILL_EXIT_PERCENT:
                            return True, f"GAP_FILLED_{fill_percent:.0f}%"
    
    # 9. STOP-LOSS CHECK (Should be handled by broker SL order, but double-check)
    stop_loss_price = entry_price * (1 - STOPLOSS_PERCENTAGE / 100)
    if current_price <= stop_loss_price:
        return True, "STOP_LOSS_HIT"
    
    return False, None

def exit_position(trader, position_id, position, exit_price, reason):
    """Execute position exit and update tracking"""
    global ACTIVE_POSITIONS, DAILY_PNL, CLOSED_POSITIONS, POSITION_PEAK_PRICES, POSITION_TRAILING_SL
    
    symbol = position.get('option_symbol') or position['symbol']
    quantity = position['quantity']
    entry_price = position['entry_price']
    instrument_key = position['instrument_key']
    
    print(f"\n{'='*120}")
    print(f"🚨 EXITING POSITION: {symbol}")
    print(f"{'='*120}")
    print(f"   Strategy: {position.get('strategy', 'UNKNOWN')}")
    print(f"   Reason: {reason}")
    print(f"   Entry Price: ₹{entry_price:.2f}")
    print(f"   Exit Price: ₹{exit_price:.2f}")
    print(f"   Quantity: {quantity}")
    
    # Calculate P&L
    pnl_per_unit = exit_price - entry_price
    total_pnl = pnl_per_unit * quantity
    pnl_percent = (pnl_per_unit / entry_price) * 100
    
    print(f"   P&L per unit: ₹{pnl_per_unit:.2f}")
    print(f"   Total P&L: ₹{total_pnl:,.2f} ({pnl_percent:+.2f}%)")
    
    try:
        # Cancel any pending SL/Target orders first
        if 'sl_order_id' in position:
            print(f"   Cancelling SL order: {position['sl_order_id']}")
            try:
                trader.cancel_order(position['sl_order_id'])
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚡ SL cancel failed: {e}")
        
        if 'target_order_id' in position:
            print(f"   Cancelling Target order: {position['target_order_id']}")
            try:
                trader.cancel_order(position['target_order_id'])
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚡ Target cancel failed: {e}")
        
        # Place market exit order
        print(f"   Placing SELL order...")
        result = trader.place_order(
            instrument_key=instrument_key,
            quantity=quantity,
            transaction_type='SELL',
            product=ORDER_PRODUCT,
            order_type='MARKET',
            price=0
        )
        
        if result.get('status_code') == 200 and result.get('response', {}).get('status') == 'success':
            exit_order_id = result['response'].get('data', {}).get('order_id')
            
            print(f"   ✅ EXIT ORDER PLACED")
            print(f"   Exit Order ID: {exit_order_id}")
            
            # Update daily P&L
            DAILY_PNL += total_pnl
            print(f"   📊 Updated Daily P&L: ₹{DAILY_PNL:,.2f}")
            
            # Store closed position details
            closed_position = {
                **position,
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'exit_reason': reason,
                'pnl': total_pnl,
                'pnl_percent': pnl_percent,
                'exit_order_id': exit_order_id
            }
            CLOSED_POSITIONS.append(closed_position)
            
            # Log the exit
            log_exit(closed_position)
            
            # Remove from active positions
            if position_id in ACTIVE_POSITIONS:
                del ACTIVE_POSITIONS[position_id]
            
            # Clear HA reversal alert state for this position
            clear_ha_alert(position_id)
            
            # Clean up peak price tracking
            if position_id in POSITION_PEAK_PRICES:
                del POSITION_PEAK_PRICES[position_id]
            if position_id in POSITION_TRAILING_SL:
                del POSITION_TRAILING_SL[position_id]
            
            print(f"{'='*120}\n")
            return True
            
        else:
            print(f"   ❌ EXIT FAILED")
            error_msg = result.get('response', {}).get('message', 'Unknown error')
            print(f"   Error: {error_msg}")
            print(f"{'='*120}\n")
            return False
            
    except Exception as e:
        print(f"   ❌ Exit error: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        print(f"{'='*120}\n")
        return False

def exit_all_positions(trader, reason="MANUAL_EXIT"):
    """Exit all active positions (emergency exit or end of day)"""
    global ACTIVE_POSITIONS
    
    if not ACTIVE_POSITIONS:
        print("ℹ️ No active positions to exit")
        return
    
    print(f"\n{'='*120}")
    print(f"🚨 EXITING ALL POSITIONS - Reason: {reason}")
    print(f"{'='*120}")
    print(f"Total positions to exit: {len(ACTIVE_POSITIONS)}")
    
    positions_to_exit = list(ACTIVE_POSITIONS.items())
    
    for position_id, position in positions_to_exit:
        instrument_key = position['instrument_key']
        current_price = trader.get_ltp(instrument_key)
        
        if current_price:
            exit_position(trader, position_id, position, current_price, reason)
            time.sleep(1)  # Brief delay between exits
        else:
            print(f"⚡ Could not get price for {position.get('symbol')}, skipping exit")
    
    print(f"{'='*120}\n")

def monitor_active_positions(trader):
    """Monitor and manage all active positions"""
    global ACTIVE_POSITIONS, DAILY_PNL, TRADING_STOPPED
    
    if not ACTIVE_POSITIONS:
        return
    
    positions_to_exit = []
    
    for position_id, position in ACTIVE_POSITIONS.items():
        instrument_key = position['instrument_key']
        
        # Get current option price
        current_price = trader.get_ltp(instrument_key)
        if not current_price:
            if DEBUG_MODE:
                print(f"⚡ Could not get price for {position.get('symbol')}")
            continue
        
        # Check exit conditions
        should_exit, reason = check_exit_conditions(position, current_price, trader)
        
        if should_exit:
            positions_to_exit.append((position_id, position, current_price, reason))
    
    # Execute exits
    for position_id, position, exit_price, reason in positions_to_exit:
        success = exit_position(trader, position_id, position, exit_price, reason)
        
        # Check if we should stop trading
        if reason in ["MAX_DAILY_LOSS", "MAX_DAILY_PROFIT"]:
            TRADING_STOPPED = True
            print(f"\n⚡ TRADING STOPPED: {reason} reached")
            print(f"Daily P&L: ₹{DAILY_PNL:,.2f}")
            exit_all_positions(trader, reason)
            break

# ═══════════════════════════════════════════════════════════════════════════════
# HEIKIN-ASHI REVERSAL ALERT — watches open positions for counter-trend signals
# ═══════════════════════════════════════════════════════════════════════════════

# Tracks which positions have already been alerted to avoid spam (one alert per
# position per HA flip event). Cleared when a position is exited.
_HA_ALERTED: set = set()   # position_id strings

# ── HA WATCHLIST FOR MISSED/REJECTED SIGNALS ──────────────────────────────────
# When a signal fires but the order is rejected (e.g. insufficient margin),
# the symbol is added here so we still watch for HA reversals and print alerts.
# Structure: { symbol: {'signal': 'LONG'|'SHORT', 'instrument_key': str,
#                        'added_at': datetime, 'reason': str} }
HA_WATCHLIST: dict = {}
HA_WATCHLIST_MAX_AGE_MINUTES = 60   # auto-expire entries older than 60 min
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ha_candles(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Convert a standard OHLCV DataFrame into Heikin-Ashi candles.
    Returns a new DataFrame with columns: ha_open, ha_high, ha_low, ha_close.
    Needs at least 3 rows.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open  = pd.Series([0.0] * n)
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, n):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high  = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low   = pd.concat([df['low'],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({
        'ha_open':  ha_open,
        'ha_high':  ha_high,
        'ha_low':   ha_low,
        'ha_close': ha_close,
    })


def _ha_colour(ha_open: float, ha_close: float) -> str:
    """Return 'green' (bullish) or 'red' (bearish) for a single HA candle."""
    return 'green' if ha_close >= ha_open else 'red'


def _ha_analyse_symbol(access_token: str, symbol: str, ikey: str, signal: str):
    """
    Shared helper: fetch 5min candles, compute HA, return analysis dict or None.
    Returns:
      { 'ha': DataFrame, 'c_prev2', 'c_prev1', 'c_last',
        'is_doji', 'is_bearish_flip', 'is_bullish_flip',
        'needs_alert', 'last_body', 'last_range', 'underlying_ltp' }
    or None if data unavailable.
    """
    try:
        df5 = fetch_5min_cached(access_token, ikey, bars=50, symbol=symbol)
        if df5 is None or len(df5) < 5:
            return None
        df5 = df5.tail(30).reset_index(drop=True)
        ha  = _compute_ha_candles(df5)
    except Exception:
        return None

    if len(ha) < 3:
        return None

    c_prev2 = _ha_colour(ha['ha_open'].iloc[-3], ha['ha_close'].iloc[-3])
    c_prev1 = _ha_colour(ha['ha_open'].iloc[-2], ha['ha_close'].iloc[-2])
    c_last  = _ha_colour(ha['ha_open'].iloc[-1], ha['ha_close'].iloc[-1])

    last_range = ha['ha_high'].iloc[-1] - ha['ha_low'].iloc[-1]
    last_body  = abs(ha['ha_close'].iloc[-1] - ha['ha_open'].iloc[-1])
    is_doji    = (last_range > 0) and (last_body / last_range < 0.20)

    is_bearish_flip = (c_prev1 == 'red'   and c_last == 'red')
    is_bullish_flip = (c_prev1 == 'green' and c_last == 'green')
    needs_alert     = (
        (signal == 'LONG'  and is_bearish_flip) or
        (signal == 'SHORT' and is_bullish_flip)
    )

    try:
        ltp_data       = get_live_prices_batch(access_token, [ikey])
        underlying_ltp = ltp_data.get(ikey, {}).get('ltp', 0) if ltp_data else 0
    except Exception:
        underlying_ltp = 0

    return {
        'ha': ha, 'c_prev2': c_prev2, 'c_prev1': c_prev1, 'c_last': c_last,
        'is_doji': is_doji, 'last_body': last_body, 'last_range': last_range,
        'is_bearish_flip': is_bearish_flip, 'is_bullish_flip': is_bullish_flip,
        'needs_alert': needs_alert, 'underlying_ltp': underlying_ltp,
    }


def _ha_klinger_check(access_token: str, ikey: str, symbol: str, signal: str):
    """
    Shared helper: check Klinger direction for HA reversal confirmation.
    Returns (klinger_confirms: bool, ko_desc: str).
    """
    try:
        kd = fetch_klinger_data_cached(access_token, ikey, symbol)
        if not kd:
            return False, "KO unavailable ⚠️"
        ko_value   = kd.get('klinger', 0)
        ko_history = kd.get('ko_history', [])

        if signal == 'LONG':   # reversal = bearish → want KO falling
            if ko_value < 0:
                return True, f"KO={ko_value:.0f} < 0 ✅"
            if len(ko_history) >= 3:
                falling = all(ko_history[i] > ko_history[i+1]
                              for i in range(len(ko_history)-3, len(ko_history)-1))
                if falling:
                    return True, (f"KO={ko_value:.0f} declining "
                                  f"{ko_history[-3]:.0f}→{ko_history[-2]:.0f}→{ko_history[-1]:.0f} ✅")
            return False, f"KO={ko_value:.0f} not yet falling ⚠️"
        else:   # SHORT reversal = bullish → want KO rising
            if ko_value > 0:
                return True, f"KO={ko_value:.0f} > 0 ✅"
            if len(ko_history) >= 3:
                rising = all(ko_history[i] < ko_history[i+1]
                             for i in range(len(ko_history)-3, len(ko_history)-1))
                if rising:
                    return True, (f"KO={ko_value:.0f} rising "
                                  f"{ko_history[-3]:.0f}→{ko_history[-2]:.0f}→{ko_history[-1]:.0f} ✅")
            return False, f"KO={ko_value:.0f} not yet rising ⚠️"
    except Exception:
        return False, "KO fetch error ⚠️"


def add_to_ha_watchlist(symbol: str, signal: str, instrument_key: str, reason: str = ""):
    """
    Add a symbol to the HA watchlist for monitoring after a rejected/missed signal.
    Called when an order cannot be placed (insufficient margin, order limit, etc.)
    """
    global HA_WATCHLIST
    HA_WATCHLIST[symbol] = {
        'signal':         signal,           # 'LONG' or 'SHORT'
        'instrument_key': instrument_key,
        'added_at':       datetime.now(),
        'reason':         reason,
    }
    print(f"👁️  HA Watchlist: added {symbol} ({signal}) — {reason}")


def check_ha_reversal_alerts(access_token: str, trader=None):
    """
    Enhancement 1 — Active positions:
      Scans ACTIVE_POSITIONS for HA colour flip + Klinger confirmation.
      When Klinger CONFIRMS the flip, auto-exits the position via exit_position()
      instead of just printing an alert.  Unconfirmed flips still print a warning.

    Enhancement 2 — Missed/rejected signals (HA_WATCHLIST):
      Monitors symbols that fired a signal but whose order was rejected (e.g. margin).
      No position exists, so no P&L — just prints a directional reversal alert
      so you can act manually if circumstances change (e.g. funds added).
      Entries auto-expire after HA_WATCHLIST_MAX_AGE_MINUTES minutes.

    Doji early-warning applies to both pools.
    Alerts fire once per symbol/position per flip event (_HA_ALERTED suppressor).
    """
    global _HA_ALERTED, HA_WATCHLIST

    # ══════════════════════════════════════════════════════════════════════════
    # PART A — ACTIVE POSITIONS (Enhancement 1: auto-exit on confirmed flip)
    # ══════════════════════════════════════════════════════════════════════════
    for pos_id, position in list(ACTIVE_POSITIONS.items()):
        symbol         = position.get('symbol', '')
        signal         = position.get('fast_trade_signal', '')
        entry_price    = position.get('entry_price', 0)
        underlying_key = position.get('underlying_key', '')

        if not symbol or signal not in ('LONG', 'SHORT'):
            continue

        ikey = underlying_key or SYMBOL_TO_ISIN.get(symbol, '')
        res  = _ha_analyse_symbol(access_token, symbol, ikey, signal)
        if res is None:
            continue

        c_prev2        = res['c_prev2']
        c_prev1        = res['c_prev1']
        c_last         = res['c_last']
        is_doji        = res['is_doji']
        needs_alert    = res['needs_alert']
        last_body      = res['last_body']
        last_range     = res['last_range']
        underlying_ltp = res['underlying_ltp']

        # ── Doji early warning ───────────────────────────────────────────────
        if is_doji and not needs_alert:
            doji_key = f"{pos_id}_doji"
            if doji_key not in _HA_ALERTED:
                _HA_ALERTED.add(doji_key)
                print(
                    f"\n⚡ HA DOJI WARNING | {symbol} | {signal} position\n"
                    f"   Last HA candle is near-doji (body={last_body:.2f} / "
                    f"range={last_range:.2f} = {last_body/last_range*100:.0f}%) "
                    f"— possible reversal forming.\n"
                    f"   HA colours: [{c_prev2}] [{c_prev1}] [{c_last}] | "
                    f"Underlying LTP: ₹{underlying_ltp:.2f} | Entry: ₹{entry_price:.2f}"
                )
            continue

        if not needs_alert:
            _HA_ALERTED.discard(f"{pos_id}_doji")
            continue

        # ── Klinger confirmation ─────────────────────────────────────────────
        klinger_confirms, ko_desc = _ha_klinger_check(access_token, ikey, symbol, signal)

        flip_key = f"{pos_id}_flip"
        if flip_key in _HA_ALERTED:
            continue

        _HA_ALERTED.add(flip_key)

        confirmed_str = "CONFIRMED ✅" if klinger_confirms else "UNCONFIRMED ⚠️ (HA flip only)"
        counter       = "SHORT/PE" if signal == 'LONG' else "LONG/CE"
        arrow         = "🔴" if signal == 'LONG' else "🟢"

        # ── NEW: Check profit and trailing stop before auto‑exit ─────────────────
        # Get current P&L % from the position (already updated by monitor_active_positions)
        pnl_pct = position.get('pnl_percent', 0.0)
        pos_abs_pnl = abs(pnl_pct)

        # Minimum profit threshold: do not auto‑exit if profit is too small
        HA_AUTO_EXIT_MIN_PROFIT_PCT = 10.0   # configurable, can be adjusted
        if klinger_confirms and trader is not None and pos_abs_pnl < HA_AUTO_EXIT_MIN_PROFIT_PCT:
            print(
                f"\n⚡ HA reversal detected but AUTO-EXIT BLOCKED for {symbol} ({signal})\n"
                f"   Reason: Profit {pnl_pct:+.1f}% < {HA_AUTO_EXIT_MIN_PROFIT_PCT}% minimum.\n"
                f"   Flip will be logged but position will stay open."
            )
            # Still print the alert (so user sees it) but do NOT exit
            print(
                f"\n{'='*80}\n"
                f"{arrow} HA REVERSAL ALERT — {symbol} | {signal} position\n"
                f"{'='*80}\n"
                f"   HA flip:       [{c_prev2}] → [{c_prev1}] → [{c_last}] "
                f"(2 consecutive {c_last} candles)\n"
                f"   Klinger:       {ko_desc}\n"
                f"   Confirmation:  {confirmed_str}\n"
                f"   Entry price:   ₹{entry_price:.2f} | Underlying LTP: ₹{underlying_ltp:.2f}\n"
                f"   Current P&L:   {pnl_pct:+.1f}% (below minimum for auto-exit)\n"
                f"   Suggestion:    Consider exiting {signal} / entering {counter} manually\n"
                f"{'='*80}"
            )
            continue

        # Block auto‑exit if trailing stop is active and drawdown is small
        # We check: trailing stop active (profit >= TRAILING_STOP_ACTIVATION) AND
        #            position has a peak price recorded, and drawdown from peak < trailing percentage.
        HA_BLOCK_WHEN_TRAILING_ACTIVE = True
        if HA_BLOCK_WHEN_TRAILING_ACTIVE and pnl_pct >= TRAILING_STOP_ACTIVATION and pos_id in POSITION_PEAK_PRICES:
            peak = POSITION_PEAK_PRICES[pos_id]
            # Drawdown from peak (percentage)
            drawdown = (peak - current_price) / peak * 100 if current_price <= peak else 0.0
            if drawdown < TRAILING_STOP_PERCENTAGE:
                print(
                    f"\n⚡ HA reversal detected but AUTO-EXIT BLOCKED for {symbol} ({signal})\n"
                    f"   Reason: Trailing stop active (profit {pnl_pct:.1f}% >= {TRAILING_STOP_ACTIVATION}%),\n"
                    f"           drawdown from peak {drawdown:.1f}% < {TRAILING_STOP_PERCENTAGE}% trail.\n"
                    f"   Letting trailing stop manage exit instead."
                )
                # Still print the alert (so user sees it) but do NOT exit
                print(
                    f"\n{'='*80}\n"
                    f"{arrow} HA REVERSAL ALERT — {symbol} | {signal} position\n"
                    f"{'='*80}\n"
                    f"   HA flip:       [{c_prev2}] → [{c_prev1}] → [{c_last}]\n"
                    f"   Klinger:       {ko_desc}\n"
                    f"   Confirmation:  {confirmed_str}\n"
                    f"   Trailing stop active — auto-exit blocked (drawdown {drawdown:.1f}% < trail)\n"
                    f"   Entry: ₹{entry_price:.2f} | Peak: ₹{peak:.2f} | Current: ₹{current_price:.2f}\n"
                    f"{'='*80}"
                )
                continue

        # ── Enhancement 1: Auto-exit on confirmed flip ───────────────────────
        if klinger_confirms and trader is not None:
            print(
                f"\n{'='*80}\n"
                f"{arrow} HA REVERSAL — AUTO-EXITING {symbol} | {signal} position\n"
                f"{'='*80}\n"
                f"   HA flip:       [{c_prev2}] → [{c_prev1}] → [{c_last}]\n"
                f"   Klinger:       {ko_desc}\n"
                f"   Confirmation:  {confirmed_str}\n"
                f"   Entry price:   ₹{entry_price:.2f} | Underlying LTP: ₹{underlying_ltp:.2f}\n"
                f"   Action:        AUTO-EXITING position now ↓\n"
                f"{'='*80}"
            )
            try:
                # Get current option price for exit
                option_key   = position.get('instrument_key', '')
                current_price = trader.get_ltp(option_key) if option_key else None
                if current_price:
                    exit_position(trader, pos_id, position, current_price,
                                  reason="HA_REVERSAL_AUTO_EXIT")
                else:
                    print(f"   ⚠️ Could not fetch option LTP for auto-exit — manual exit needed.")
            except Exception as e:
                print(f"   ⚠️ Auto-exit error: {e} — manual exit needed.")
        else:
            # Unconfirmed flip or no trader — print alert only
            print(
                f"\n{'='*80}\n"
                f"{arrow} HA REVERSAL ALERT — {symbol} | {signal} position\n"
                f"{'='*80}\n"
                f"   HA flip:       [{c_prev2}] → [{c_prev1}] → [{c_last}] "
                f"(2 consecutive {c_last} candles)\n"
                f"   Klinger:       {ko_desc}\n"
                f"   Confirmation:  {confirmed_str}\n"
                f"   Entry price:   ₹{entry_price:.2f} | Underlying LTP: ₹{underlying_ltp:.2f}\n"
                f"   Suggestion:    Consider exiting {signal} / entering {counter}\n"
                f"{'='*80}"
            )
            if not klinger_confirms:
                print(f"   ℹ️  Klinger not yet confirming — watch next 1-2 candles before acting.")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B — HA WATCHLIST: missed/rejected signals (Enhancement 2)
    # ══════════════════════════════════════════════════════════════════════════
    now = datetime.now()
    expired = [sym for sym, w in HA_WATCHLIST.items()
               if (now - w['added_at']).total_seconds() / 60 > HA_WATCHLIST_MAX_AGE_MINUTES]
    for sym in expired:
        del HA_WATCHLIST[sym]
        _HA_ALERTED.discard(f"wl_{sym}_flip")
        _HA_ALERTED.discard(f"wl_{sym}_doji")
        if DEBUG_MODE:
            print(f"👁️  HA Watchlist: {sym} expired (>{HA_WATCHLIST_MAX_AGE_MINUTES}min)")

    for symbol, entry in list(HA_WATCHLIST.items()):
        signal = entry['signal']
        ikey   = entry['instrument_key'] or SYMBOL_TO_ISIN.get(symbol, '')
        reason = entry['reason']
        added  = entry['added_at'].strftime('%H:%M')

        res = _ha_analyse_symbol(access_token, symbol, ikey, signal)
        if res is None:
            continue

        c_prev2        = res['c_prev2']
        c_prev1        = res['c_prev1']
        c_last         = res['c_last']
        is_doji        = res['is_doji']
        needs_alert    = res['needs_alert']
        last_body      = res['last_body']
        last_range     = res['last_range']
        underlying_ltp = res['underlying_ltp']

        wl_id = f"wl_{symbol}"   # watchlist entries use symbol as key (no pos_id)

        # ── Doji early warning ───────────────────────────────────────────────
        if is_doji and not needs_alert:
            doji_key = f"{wl_id}_doji"
            if doji_key not in _HA_ALERTED:
                _HA_ALERTED.add(doji_key)
                print(
                    f"\n⚡ HA DOJI WARNING (WATCHLIST) | {symbol} | missed {signal} @ {added}\n"
                    f"   Rejection reason: {reason}\n"
                    f"   Last HA candle near-doji (body={last_body:.2f} / "
                    f"range={last_range:.2f} = {last_body/last_range*100:.0f}%) "
                    f"— possible reversal forming.\n"
                    f"   HA colours: [{c_prev2}] [{c_prev1}] [{c_last}] | "
                    f"Underlying LTP: ₹{underlying_ltp:.2f}"
                )
            continue

        if not needs_alert:
            _HA_ALERTED.discard(f"{wl_id}_doji")
            continue

        # ── Klinger confirmation ─────────────────────────────────────────────
        klinger_confirms, ko_desc = _ha_klinger_check(access_token, ikey, symbol, signal)

        flip_key = f"{wl_id}_flip"
        if flip_key in _HA_ALERTED:
            continue

        _HA_ALERTED.add(flip_key)

        confirmed_str = "CONFIRMED ✅" if klinger_confirms else "UNCONFIRMED ⚠️ (HA flip only)"
        counter       = "SHORT/PE" if signal == 'LONG' else "LONG/CE"
        arrow         = "🔴" if signal == 'LONG' else "🟢"

        print(
            f"\n{'='*80}\n"
            f"{arrow} HA REVERSAL ALERT (WATCHLIST) — {symbol} | missed {signal} signal @ {added}\n"
            f"{'='*80}\n"
            f"   Rejection:     {reason}\n"
            f"   HA flip:       [{c_prev2}] → [{c_prev1}] → [{c_last}] "
            f"(2 consecutive {c_last} candles)\n"
            f"   Klinger:       {ko_desc}\n"
            f"   Confirmation:  {confirmed_str}\n"
            f"   Underlying LTP: ₹{underlying_ltp:.2f}\n"
            f"   Note:          No position open — informational only.\n"
            f"{'='*80}"
        )
        if not klinger_confirms:
            print(f"   ℹ️  Klinger not yet confirming — watch next 1-2 candles.")


def clear_ha_alert(position_id: str):
    """Call when a position is exited — clears its HA alert state."""
    _HA_ALERTED.discard(f"{position_id}_flip")
    _HA_ALERTED.discard(f"{position_id}_doji")


def remove_from_ha_watchlist(symbol: str):
    """Remove a symbol from the HA watchlist (e.g. when funds are added and order succeeds)."""
    global HA_WATCHLIST
    if symbol in HA_WATCHLIST:
        del HA_WATCHLIST[symbol]
        _HA_ALERTED.discard(f"wl_{symbol}_flip")
        _HA_ALERTED.discard(f"wl_{symbol}_doji")
        if DEBUG_MODE:
            print(f"👁️  HA Watchlist: {symbol} removed")

# ═══════════════════════════════════════════════════════════════════════════════
# END HEIKIN-ASHI REVERSAL ALERT
# ═══════════════════════════════════════════════════════════════════════════════


def log_exit(closed_position):
    """Log exit details to CSV and text file"""
    try:
        # CSV log
        with open(EXIT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                closed_position['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                closed_position['symbol'],
                closed_position.get('strategy', 'UNKNOWN'),
                closed_position['entry_price'],
                closed_position['exit_price'],
                closed_position['quantity'],
                closed_position['pnl'],
                closed_position['pnl_percent'],
                closed_position['exit_reason'],
                closed_position.get('order_id', 'N/A')
            ])
        
        # Text log
        with open(EXIT_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"EXIT: {closed_position['symbol']}\n")
            f.write(f"Time: {closed_position['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Strategy: {closed_position.get('strategy', 'UNKNOWN')}\n")
            f.write(f"Entry: ₹{closed_position['entry_price']:.2f} | Exit: ₹{closed_position['exit_price']:.2f}\n")
            f.write(f"P&L: ₹{closed_position['pnl']:,.2f} ({closed_position['pnl_percent']:+.2f}%)\n")
            f.write(f"Reason: {closed_position['exit_reason']}\n")
            f.write(f"{'='*100}\n")
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚡ Exit logging error: {e}")

def sync_positions_with_broker(trader):
    """Sync active positions with actual broker positions"""
    global ACTIVE_POSITIONS
    
    try:
        broker_positions = trader.get_positions()
        
        if broker_positions.get('status') != 'success':
            return
        
        positions_data = broker_positions.get('data', [])
        
        # Get instrument keys from broker positions
        broker_keys = set()
        for pos in positions_data:
            if pos.get('quantity', 0) != 0:  # Only open positions
                broker_keys.add(pos.get('instrument_token'))
        
        # Check if any of our tracked positions are no longer open
        closed_keys = []
        for position_id, position in ACTIVE_POSITIONS.items():
            if position['instrument_key'] not in broker_keys:
                # Position was closed outside our system (manual or SL hit)
                closed_keys.append(position_id)
        
        # Handle positions closed outside system
        for position_id in closed_keys:
            position = ACTIVE_POSITIONS[position_id]
            print(f"\n⚡ Position closed outside system: {position.get('symbol')}")
            
            # Try to get exit price from order history
            exit_price = position['entry_price']  # Default
            
            closed_position = {
                **position,
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'exit_reason': 'EXTERNAL_CLOSE',
                'pnl': 0,
                'pnl_percent': 0
            }
            CLOSED_POSITIONS.append(closed_position)
            log_exit(closed_position)
            
            del ACTIVE_POSITIONS[position_id]
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚡ Position sync error: {e}")

# ========== ORDER PLACEMENT FUNCTIONS ==========
def verify_order_result(trader, result, symbol):
    """Verify order placement result - FIXED VERSION"""
    if not result:
        print("❌ No order placed")
        return None
    
    status_code = result.get('status_code')
    response = result.get('response', {})
    
    print(f"\n📨 API Response:")
    print(f" Status Code: {status_code}")
    print(f" Status: {response.get('status')}")
    
    # FIX 2: Hard stop if order API failed
    if status_code != 200:
        print(f"❌ ORDER API FAILED (Status {status_code}): {response}")
        error_message = response.get('message', response.get('errors', 'Unknown error'))
        print(f"❌ Rejection Reason: {error_message}")
        return None
    
    if status_code == 200 and response.get('status') == 'success':
        order_id = response.get('data', {}).get('order_id')
        if order_id:
            print(f"\n🎉 ORDER PLACED SUCCESSFULLY!")
            print(f" Order ID: {order_id}")
            print(f"\n ⏳ Waiting {ORDER_VERIFICATION_DELAY} seconds...")
            time.sleep(ORDER_VERIFICATION_DELAY)
            print("\n 🔍 Checking order status...")
            order_details = trader.get_order_details(order_id)
            if order_details.get('status') == 'success' and 'data' in order_details:
                for order in order_details['data']:
                    status = order.get('status')
                    print(f"\n 📋 Order Status: {status.upper()}")
                    print(f" Symbol: {order.get('tradingsymbol')}")
                    print(f" Qty: {order.get('quantity')}")
                    avg_price = order.get('average_price', 0)
                    if avg_price > 0:
                        print(f" Filled Price: ₹{avg_price:.2f}")
                        total = avg_price * order.get('quantity', 0)
                        print(f" Total Value: ₹{total:,.2f}")
                    if status == 'complete':
                        print("\n ✅ ORDER EXECUTED!")
                        return {
                            'order_id': order_id,
                            'status': 'complete',
                            'filled_price': avg_price,
                            'quantity': order.get('quantity')
                        }
                    elif status == 'rejected':
                        print(f"\n ❌ REJECTED: {order.get('status_message')}")
                        return None
                    elif status in ['pending', 'open pending', 'trigger pending']:
                        print("\n ⏳ Order is PENDING")
                        return {
                            'order_id': order_id,
                            'status': 'pending',
                            'quantity': order.get('quantity')
                        }
            return {'order_id': order_id, 'status': 'unknown'}
        else:
            print("\n❌ ORDER FAILED")
            error_msg = response.get('message', 'Unknown error')
            print(f" Error: {error_msg}")
            if 'errors' in response:
                for error in response['errors']:
                    msg = error.get('message', error) if isinstance(error, dict) else str(error)
                    print(f" • {msg}")
            return None
    else:
        print("\n❌ ORDER FAILED")
        error_msg = response.get('message', 'Unknown error')
        print(f" Error: {error_msg}")
        return None

def place_breakout_order(breakout_data, trader):
    """Place OPTION orders for R3/S3/Box/Range breakouts"""
    global DAILY_ORDER_COUNT, LAST_ORDER_TIME
    global PLACED_ORDERS, ACTIVE_POSITIONS, TRADING_STOPPED, POSITION_TRAILING_SL

    if TRADING_STOPPED:
        print("⚡ Trading stopped - no new orders")
        return None

    # Guard: Upstox rejects orders outside 05:30–23:59 with HTTP 423.
    # Check here (before option-chain lookup) to avoid wasted API calls.
    if not is_order_time_allowed():
        symbol   = breakout_data.get('symbol', '?')
        strategy = breakout_data.get('strategy', 'UNKNOWN')
        print(f"⏭️  {symbol} {strategy}: order skipped — outside Upstox service hours (05:30–23:59 IST)")
        return None

    symbol = breakout_data['symbol']
    underlying_key = breakout_data['instrument_key']
    breakout_type = breakout_data.get('breakout_type', 'CE')
    strategy = breakout_data.get('strategy', 'UNKNOWN')

    option_type = 'CE' if breakout_type == 'CE' else 'PE'

    # Select option contract
    selection = select_liquid_stock_option_contract(
        trader=trader,
        underlying_key=underlying_key,
        symbol=symbol,
        option_type=option_type,
        strike_offset_steps=0,
    )
    if not selection:
        print(f"⚡ Skipping {strategy} trade in {symbol} - no suitable {option_type} option.")
        return None

    option_key, option_symbol, strike, lot_size, premium, contract, is_premium_estimated = selection
    total_qty = lot_size * ORDER_QUANTITY

    print(f"\n📊 PLACING {option_type} OPTION ORDER for {symbol} ({strategy})")
    print(f" Underlying: {symbol}")
    print(f" Option: {option_symbol}")
    print(f" Strike: {strike} | Expiry: {contract.get('expiry')}")
    print(f" Lots: {ORDER_QUANTITY} | Lot size: {lot_size} | Total Qty: {total_qty}")
    print(f" Approx premium: ₹{premium:.2f} ({'ESTIMATED' if is_premium_estimated else 'LTP'})")
    if breakout_data.get('underlying_stop_loss'):
        print(
            f" Underlying setup: entry ₹{breakout_data.get('current_price', 0):.2f} | "
            f"stop ₹{breakout_data.get('underlying_stop_loss', 0):.2f} | "
            f"target ₹{breakout_data.get('underlying_target', 0):.2f}"
        )
    
    # Show Klinger status if available
    if breakout_data.get('klinger_status'):
        print(f" 🔥 Klinger: {breakout_data['klinger_status']}")

    try:
        # FIX 6: Use LIMIT orders instead of MARKET for options
        limit_price = round(premium * 1.02, 2)  # Add 2% buffer for slippage
        
        print(f" 💰 Order Type: LIMIT @ ₹{limit_price:.2f} (premium ₹{premium:.2f} + 2% buffer)")
        
        result = trader.place_order(
            instrument_key=option_key,
            quantity=total_qty,
            transaction_type='BUY',
            product=ORDER_PRODUCT,
            order_type='LIMIT',      # <-- changed from MARKET
            price=limit_price         # <-- limit price added
        )

        order_info = verify_order_result(trader, result, option_symbol)
        if order_info and order_info.get('order_id'):
            order_id = order_info['order_id']
            
            # Update counters based on strategy
            DAILY_ORDER_COUNT += 1
                
            LAST_ORDER_TIME[symbol] = datetime.now()
            filled_price = order_info.get('filled_price', premium)

            # Parse expiry date
            expiry_date = None
            try:
                expiry_date = datetime.strptime(contract.get('expiry'), "%Y-%m-%d")
            except:
                pass

            # Create position record
            position_record = {
                'order_id': order_id,
                'symbol': symbol,
                'option_symbol': option_symbol,
                'instrument_key': option_key,
                'underlying_key': underlying_key,
                'entry_price': filled_price,
                'quantity': total_qty,
                'breakout_type': breakout_type,
                'option_type': option_type,
                'trade_type': f'{strategy}_OPTION',
                'strategy': strategy,
                'timestamp': datetime.now(),
                'expiry_date': expiry_date,
                'klinger_confirmed': breakout_data.get('klinger_confirmed', False),
                'is_premium_estimated': is_premium_estimated,
                'signal_entry_price': breakout_data.get('current_price'),
                'entry_trigger': breakout_data.get('entry_trigger'),
                'pullback_low': breakout_data.get('pullback_low'),
                'pullback_rsi_2': breakout_data.get('pullback_rsi_2'),
                'current_rsi_2': breakout_data.get('current_rsi_2'),
                'vwap_at_entry': breakout_data.get('vwap_value'),
                'ema_200_at_entry': breakout_data.get('ema_200'),
                'underlying_stop_loss': breakout_data.get('underlying_stop_loss'),
                'underlying_target': breakout_data.get('underlying_target'),
                'risk_per_share': breakout_data.get('risk_per_share'),
                'trail_activation_price': breakout_data.get('trail_activation_price'),
                'dynamic_trail_ema': breakout_data.get('dynamic_trail_ema'),
                'rr_target_hit': False
            }

            PLACED_ORDERS[order_id] = position_record
            
            # Add to active positions for exit management
            if order_info.get('status') == 'complete':
                ACTIVE_POSITIONS[order_id] = position_record.copy()

            # Stop-loss
            if PLACE_STOPLOSS and order_info.get('status') == 'complete':
                safety_sl_pct = PULLBACK_OPTION_SAFETY_SL_PERCENT if strategy == 'PULLBACK_CE' else STOPLOSS_PERCENTAGE
                sl_trigger = round(filled_price * (1 - safety_sl_pct / 100), 2)
                sl_limit = round(sl_trigger * 0.99, 2)

                print("\n🛡️ PLACING OPTION STOP-LOSS")
                print(f" Trigger: ₹{sl_trigger:.2f} | Limit: ₹{sl_limit:.2f}")
                if strategy == 'PULLBACK_CE':
                    print(" Structural stop/target are enforced on the underlying; this premium SL is a safety fallback.")

                try:
                    sl_result = trader.place_order(
                        instrument_key=option_key,
                        quantity=total_qty,
                        transaction_type='SELL',
                        product=ORDER_PRODUCT,
                        order_type='SL_LIMIT',   # <-- changed from 'SL'
                        price=sl_limit,
                        trigger_price=sl_trigger
                    )
                    if sl_result.get('status_code') == 200:
                        sl_order_id = sl_result['response'].get('data', {}).get('order_id')
                        if sl_order_id:
                            print(f"✅ SL Order ID: {sl_order_id}")
                            PLACED_ORDERS[order_id]['sl_order_id'] = sl_order_id
                            ACTIVE_POSITIONS[order_id]['sl_order_id'] = sl_order_id
                            POSITION_TRAILING_SL[order_id] = sl_trigger   # seed initial level
                except Exception as e:
                    print(f"⚡ SL placement error: {e}")

            return order_id

    except Exception as e:
        print(f"❌ Order error: {e}")
        return None

def send_alert(b, trader=None):
    """Send R3/S3/Box/Range breakout alerts"""
    global DAILY_ORDER_COUNT, LAST_ORDER_TIME
    
    s = b['symbol']
    strategy = b.get('strategy', 'UNKNOWN')
    
    # ── ALERT REGISTRATION ────────────────────────────────────────────────────
    if strategy in ['R3', 'PULLBACK_CE']:
        if s in R3_ALERTED_STOCKS:
            return
        R3_ALERTED_STOCKS.add(s)
        ALERTED_STOCKS.add(s)
        csv_file = ALERT_CSV_FILE
    elif strategy == 'S3':
        if s in S3_ALERTED_STOCKS:
            return
        S3_ALERTED_STOCKS.add(s)
        ALERTED_STOCKS.add(s)
        csv_file = ALERT_CSV_FILE
    else:
        if s in ALERTED_STOCKS:
            return
        ALERTED_STOCKS.add(s)
        csv_file = ALERT_CSV_FILE
    
    day_gain = 100.0*(b['current_price'] - b['yesterday_close'])/b['yesterday_close'] if b['yesterday_close'] else 0.0
    
    is_pe = b.get('breakout_type') == 'PE'
    
    print("\n" + "="*120)
    print(f"🚀 {strategy} {'BREAKDOWN' if is_pe else 'BREAKOUT'} ALERT! 🚀")
    print("="*120)
    
    print(f"Stock: {s} | Time: {b['timestamp'].strftime('%H:%M:%S')}")
    print(f"Level: ₹{b.get('level', b.get('r3', b.get('s3', 0))):.2f} | Price: ₹{b['current_price']:.2f}")
    print(f"Volume: {b['volume_ratio']:.2f}x | Day Gain: {day_gain:+.2f}%")
    print(f"Strategy: {strategy} | Type: {b['breakout_type']}")
    if strategy == 'PULLBACK_CE':
        current_rsi = b.get('current_rsi_2')
        current_rsi_str = f"{current_rsi:.1f}" if isinstance(current_rsi, (int, float)) else "N/A"
        print(
            f"Trigger: {b.get('entry_trigger', 'N/A')} | Pullback RSI(2): {b.get('pullback_rsi_2', 0):.1f} | "
            f"Current RSI(2): {current_rsi_str}"
        )
        print(
            f"VWAP: ₹{b.get('vwap_value', 0):.2f} | EMA{TREND_FILTER_EMA_PERIOD}: ₹{b.get('ema_200', 0):.2f} | "
            f"Stop: ₹{b.get('underlying_stop_loss', 0):.2f} | 2R Target: ₹{b.get('underlying_target', 0):.2f}"
        )
    
    # Show Klinger confirmation if available
    if b.get('klinger_confirmed'):
        print(f"🔥 Klinger: {b.get('klinger_status', 'CONFIRMED')}")
    
    # Automated order placement
    if ENABLE_AUTO_TRADING and trader and not TRADING_STOPPED:
        print("\n" + "-"*120)
        total_orders = DAILY_ORDER_COUNT
        if total_orders >= MAX_ORDERS_PER_DAY:
            print("⚡ Daily order limit reached")
        elif s in LAST_ORDER_TIME:
            time_since = (datetime.now() - LAST_ORDER_TIME[s]).seconds
            if time_since < MIN_ORDER_GAP_SECONDS:
                print(f"⚡ Too soon ({time_since}s)")
            else:
                place_breakout_order(b, trader)
        else:
            place_breakout_order(b, trader)
        print("-" * 120)

    print("="*120 + "\n")
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            s,
            b.get('breakout_type', 'CE'),
            f"{b['current_price']:.2f}",
            f"{b['volume_ratio']:.2f}",
            f"{day_gain:.2f}%",
            strategy,
            b.get('klinger_status', 'N/A')
        ])

def print_position_summary():
    """Print summary of active positions"""
    global ACTIVE_POSITIONS, DAILY_PNL
    
    if not ACTIVE_POSITIONS:
        return
    
    print(f"\n{'='*120}")
    print(f"📊 ACTIVE POSITIONS SUMMARY ({len(ACTIVE_POSITIONS)} positions)")
    print(f"{'='*120}")
    
    total_unrealized_pnl = 0
    
    for position_id, position in ACTIVE_POSITIONS.items():
        symbol = position.get('option_symbol') or position['symbol']
        strategy = position.get('strategy', 'UNKNOWN')
        entry_price = position['entry_price']
        current_pnl = position.get('current_pnl', 0)
        pnl_percent = position.get('pnl_percent', 0)
        
        total_unrealized_pnl += current_pnl
        
        print(f"• {symbol} ({strategy})")
        print(f"  Entry: ₹{entry_price:.2f} | P&L: ₹{current_pnl:,.2f} ({pnl_percent:+.2f}%)")
    
    print(f"\n💰 Daily Realized P&L: ₹{DAILY_PNL:,.2f}")
    print(f"📈 Unrealized P&L: ₹{total_unrealized_pnl:,.2f}")
    print(f"📊 Total P&L: ₹{DAILY_PNL + total_unrealized_pnl:,.2f}")
    print(f"{'='*120}\n")

def print_final_stats():
    """Print session summary statistics"""
    global CLOSED_POSITIONS
    
    print(f"\n{'='*100}")
    print("⚡ TRADING SESSION COMPLETE")
    print(f"{'='*100}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total Stocks Monitored: {len(R3_LEVELS)}")
    print(f"\n📊 ALERTS SUMMARY:")
    print(f" • Directional Alerts: {len(ALERTED_STOCKS)} (LONG={len(R3_ALERTED_STOCKS)}, SHORT={len(S3_ALERTED_STOCKS)})")
    if ENABLE_ORB_STRATEGY:
        print(f" • ORB Alerts: {len(ORB_ALERTED_STOCKS)}")
    
    print(f"\n📈 ORDERS SUMMARY:")
    print(f" • Directional Orders: {DAILY_ORDER_COUNT}")
    if ENABLE_ORB_STRATEGY:
        print(f" • ORB Orders: {ORB_ORDER_COUNT}")
    total_orders = DAILY_ORDER_COUNT + ORB_ORDER_COUNT
    print(f" • Total Orders: {total_orders}")
    
    if ENABLE_KLINGER_FILTER:
        klinger_confirmed_orders = sum(1 for order in PLACED_ORDERS.values() 
                                       if order.get('klinger_confirmed', False))
        print(f"\n🔥 KLINGER STATS:")
        print(f" • Orders with Klinger confirmation: {klinger_confirmed_orders}")
        if len(PLACED_ORDERS) > 0:
            print(f" • Klinger confirmation rate: {(klinger_confirmed_orders/len(PLACED_ORDERS)*100):.1f}%")
        if KLINGER_PAPER_MODE:
            print(f" • ✗ PAPER MODE: Logging only, not filtering")

    # Rejected signals summary
    if REJECTED_ORDER_SIGNALS:
        from collections import Counter
        reasons = Counter(r['reason'] for r in REJECTED_ORDER_SIGNALS)
        print(f"\n⚠️ REJECTED SIGNALS (signals fired but order not placed): {len(REJECTED_ORDER_SIGNALS)}")
        for reason, count in reasons.most_common():
            print(f"   • {reason}: {count} signal(s)")
    
    # Cache Stats
    if ENABLE_CANDLE_CACHE:
        print_cache_statistics()
    
    # Exit Management Stats
    print(f"\n🚨 EXIT MANAGEMENT STATS:")
    print(f" • Total Closed Positions: {len(CLOSED_POSITIONS)}")
    print(f" • Active Positions: {len(ACTIVE_POSITIONS)}")
    print(f" • Daily Realized P&L: ₹{DAILY_PNL:,.2f}")
    
    if CLOSED_POSITIONS:
        winning_trades = [p for p in CLOSED_POSITIONS if p['pnl'] > 0]
        losing_trades = [p for p in CLOSED_POSITIONS if p['pnl'] < 0]
        
        print(f" • Winning Trades: {len(winning_trades)}")
        print(f" • Losing Trades: {len(losing_trades)}")
        
        if len(CLOSED_POSITIONS) > 0:
            win_rate = (len(winning_trades) / len(CLOSED_POSITIONS)) * 100
            print(f" • Win Rate: {win_rate:.1f}%")
            
            avg_win = sum(p['pnl'] for p in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(p['pnl'] for p in losing_trades) / len(losing_trades) if losing_trades else 0
            
            print(f" • Avg Win: ₹{avg_win:,.2f}")
            print(f" • Avg Loss: ₹{avg_loss:,.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
                print(f" • Profit Factor: {profit_factor:.2f}")
        
        # Exit reasons breakdown
        exit_reasons = {}
        for pos in CLOSED_POSITIONS:
            reason = pos['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        print(f"\n 📋 Exit Reasons Breakdown:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {reason}: {count}")
    
    # ORB Stats
    if ENABLE_ORB_STRATEGY:
        print_orb_summary()
    
    if PLACED_ORDERS:
        print(f"\n📋 ALL ORDERS PLACED:")
        for order_id, order_info in PLACED_ORDERS.items():
            trade_type = order_info.get('trade_type', 'Unknown')
            symbol = order_info['symbol']
            strategy = order_info.get('strategy', '')
            klinger = "✓" if order_info.get('klinger_confirmed') else ""
            premium_source = "EST" if order_info.get('is_premium_estimated') else "LTP"
            print(f" • {symbol} ({trade_type} {strategy}) {klinger}[{premium_source}]: Order {order_id}")
            print(f"   Entry: ₹{order_info['entry_price']:.2f} | Qty: {order_info['quantity']}")
            if 'sl_order_id' in order_info:
                print(f"   SL: {order_info['sl_order_id']}")
            if 'target_order_id' in order_info:
                print(f"   Target: {order_info['target_order_id']}")
    
    if ACTIVE_POSITIONS:
        print(f"\n⚡ OPEN POSITIONS (Not Exited):")
        for position_id, position in ACTIVE_POSITIONS.items():
            symbol = position.get('option_symbol') or position['symbol']
            print(f" • {symbol}: {position.get('strategy', 'UNKNOWN')}")
            print(f"   Entry: ₹{position['entry_price']:.2f} | Current P&L: ₹{position.get('current_pnl', 0):,.2f}")
    
    print(f"{'='*100}\n")
    
    # Final log
    with open(ALERT_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"SESSION END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directional Alerts: {len(ALERTED_STOCKS)} | Orders: {DAILY_ORDER_COUNT}\n")
        if ENABLE_ORB_STRATEGY:
            f.write(f"ORB Alerts: {len(ORB_ALERTED_STOCKS)} | Orders: {ORB_ORDER_COUNT}\n")
        f.write(f"Total Orders: {DAILY_ORDER_COUNT + ORB_ORDER_COUNT}\n")
        f.write(f"Closed Positions: {len(CLOSED_POSITIONS)} | Daily P&L: ₹{DAILY_PNL:,.2f}\n")
        if ENABLE_KLINGER_FILTER:
            klinger_count = sum(1 for order in PLACED_ORDERS.values() 
                              if order.get('klinger_confirmed', False))
            f.write(f"Klinger Confirmed Orders: {klinger_count}\n")
        if ENABLE_CANDLE_CACHE:
            f.write(f"Cache Hits: {CACHE_STATS['cache_hits']} | Misses: {CACHE_STATS['cache_misses']}\n")
        f.write(f"{'='*100}\n\n")

# ========== MAIN MONITORING LOOP ==========
def enhanced_monitor(access_token, keys, symbols):
    """Main monitoring loop with all strategies including Klinger, Caching, and Exit Management"""
    print("📡 STARTING REAL-TIME MONITORING WITH INTELLIGENT CACHING...")

    # ── STARTUP ORDER-WINDOW WAIT ─────────────────────────────────────────────
    # Upstox rejects orders with HTTP 423 before 05:30 IST.
    # When TEST_MODE=True and WAIT_FOR_ORDER_WINDOW=True, wait here rather than
    # running scans that generate signals whose orders will all be rejected.
    # Set WAIT_FOR_ORDER_WINDOW=False to scan immediately (signals logged, no orders).
    if WAIT_FOR_ORDER_WINDOW and not is_order_time_allowed():
        now       = datetime.now()
        opens_at  = now.replace(hour=5, minute=30, second=0, microsecond=0)
        if now > opens_at:          # already past 05:30 → means we're after midnight
            opens_at = opens_at + timedelta(days=1)
        wait_secs = int((opens_at - now).total_seconds())
        wait_mins = wait_secs // 60
        print(f"\n⏳ ORDER WINDOW WAIT: Upstox API opens at 05:30 IST.")
        print(f"   Current time: {now.strftime('%H:%M:%S')} | Opens in: {wait_mins}m {wait_secs%60}s")
        print(f"   (Set WAIT_FOR_ORDER_WINDOW=False to skip this wait and scan immediately)\n")
        while not is_order_time_allowed():
            remaining = int((opens_at - datetime.now()).total_seconds())
            if remaining % 300 == 0 or remaining <= 60:  # print every 5 min + last minute
                print(f"   ⏳ Waiting for 05:30... {remaining//60}m {remaining%60}s remaining",
                      flush=True)
            time.sleep(30)
        print(f"\n✅ Order window open ({datetime.now().strftime('%H:%M:%S')}) — starting scans.\n")
    # ─────────────────────────────────────────────────────────────────────────
    print(f"🔥 Klinger Filter: {'ENABLED ✓' if ENABLE_KLINGER_FILTER else 'DISABLED'}")
    if ENABLE_KLINGER_FILTER:
        print(f"   • Box Trading: {'ON' if ENABLE_KLINGER_FOR_BOX else 'OFF'}")
        print(f"   • Range Trading: {'ON' if ENABLE_KLINGER_FOR_RANGE else 'OFF'}")
        print(f"   • Adaptive Mode: {'ON' if ADAPTIVE_KLINGER_LOOKBACK else 'OFF'}")
        if KLINGER_PAPER_MODE:
            print(f"   • ✗ PAPER MODE: Logging only, not filtering")
    print(f"💾 Candle Cache: {'ENABLED ✓' if ENABLE_CANDLE_CACHE else 'DISABLED'}")
    print(f"📊 ORB Strategy: {'ENABLED ✓' if ENABLE_ORB_STRATEGY else 'DISABLED'}")
    print(f"🚨 Exit Management: {'ENABLED ✓' if ENABLE_EXIT_MANAGEMENT else 'DISABLED'}")
    print(f"Confirmation cycles required: {BREACH_CONFIRMATION_CYCLES}")
    print(f"Time window: {BREACH_TIME_WINDOW}s")
    print(f"Price sustainability: {PRICE_SUSTAINABILITY_PERCENT}%")
    print("Scanning every 30 seconds...\n")
    
    trader = None
    if ENABLE_AUTO_TRADING:
        trader = UpstoxTrader(access_token)

    # ── Start AI assistant (background thread, Groq free API) ─────────────────
    print(f"\n{ai_status()}")
    start_ai_assistant(lambda: globals(), lambda: trader)
    # ─────────────────────────────────────────────────────────────────────────
    
    scan_count = 0
    klinger_update_batch = []
    last_position_check = datetime.now()
    last_position_sync = datetime.now()
    last_summary_print = datetime.now()
    last_cache_cleanup = datetime.now()
    
    try:
        # Initialize FII/DII and ORB
        if ENABLE_FII_DII_FILTER:
            print("\n🔍 Initializing FII/DII data...")
            extract_fii_dii_data()
        
        if ENABLE_ORB_STRATEGY:
            initialize_orb_csv_files()

            # ── ORB STARTUP CATCHUP ───────────────────────────────────────────
            # If the bot starts between 09:20 and the breakout window close
            # (default 09:50), run the primary ORB pass immediately — before
            # the first scan fires. This ensures signals are built on the very
            # first scan rather than one 30-second cycle later.
            # check_orb_time_and_process handles the same logic on every scan,
            # but calling it here with empty live_data is harmless (it won't
            # process anything useful without live prices). We just prime the
            # state so scan #1 doesn't need to re-trigger the primary pass.
            _now = datetime.now()
            _920 = _now.replace(hour=9, minute=20, second=0, microsecond=0)
            _cutoff = _920 + timedelta(minutes=ORB_BREAKOUT_WINDOW_MINUTES)
            _ct = _now.strftime("%H:%M")
            if ENABLE_ORB_STRATEGY and _ct >= "09:20" and _now < _cutoff and not ORB_PROCESSED_TODAY:
                print(f"\n🕘 ORB startup: bot started at {_ct}, within ORB window "
                      f"(09:20–{_cutoff.strftime('%H:%M')}). "
                      f"Will run primary ORB pass on first live-data scan.")
            # ── END ORB STARTUP CATCHUP ──────────────────────────────────────
        
        while True:
            scan_count += 1
            current_time = datetime.now()

            # Clear intraday candle cache from previous scan cycle
            clear_intraday_cache()
            
            # Heartbeat - always visible so you know the bot is alive
            print(f"\n🔄 Scan #{scan_count} | {current_time.strftime('%H:%M:%S')} | Orders today: {DAILY_ORDER_COUNT+ORB_ORDER_COUNT}", flush=True)
            
            # Check if trading should be stopped
            if TRADING_STOPPED:
                print(f"\n⚡ Trading stopped. Reason: Daily limit reached")
                print("Continuing to monitor positions for exits...")
            
            # Update Klinger every 5 scans (2.5 minutes)
            if ENABLE_KLINGER_FILTER and scan_count % 5 == 0:
                if not klinger_update_batch:
                    # Prepare batch of stocks to update (top 30 by volume)
                    sorted_keys = sorted(
                        R3_LEVELS.keys(),
                        key=lambda k: R3_LEVELS[k].get('avg_volume_20d', 0),
                        reverse=True
                    )[:30]
                    klinger_update_batch = sorted_keys.copy()
                    if DEBUG_MODE:
                        print(f"\n📊 Preparing to update Klinger for {len(klinger_update_batch)} stocks...")
                
                # Update 5 stocks per scan to avoid blocking
                updated_count = 0
                for _ in range(min(5, len(klinger_update_batch))):
                    if klinger_update_batch:
                        key_to_update = klinger_update_batch.pop(0)
                        symbol = R3_LEVELS[key_to_update].get('symbol')
                        if symbol:
                            # Use cached Klinger update
                            klinger_data = fetch_klinger_data_cached(access_token, key_to_update, symbol)
                            if klinger_data:
                                R3_LEVELS[key_to_update]['klinger'] = klinger_data
                                updated_count += 1
                
                if DEBUG_MODE and updated_count > 0:
                    print(f"✓ Updated Klinger for {updated_count} stocks | Remaining: {len(klinger_update_batch)}")
            
            # Clean up stale states every 10 scans
            if scan_count % 10 == 0:
                reset_stale_breach_states()
            
            # Cache cleanup every hour
            if ENABLE_CANDLE_CACHE:
                time_since_cleanup = (current_time - last_cache_cleanup).seconds
                if time_since_cleanup >= 3600:  # 1 hour
                    print("\n🧹 Running cache cleanup...")
                    cleanup_old_cache()
                    save_cache_stats()
                    last_cache_cleanup = current_time
            
            # Monitor active positions (every POSITION_MONITORING_INTERVAL seconds)
            if ENABLE_EXIT_MANAGEMENT and trader and ACTIVE_POSITIONS:
                time_since_check = (current_time - last_position_check).seconds
                if time_since_check >= POSITION_MONITORING_INTERVAL:
                    if DEBUG_MODE:
                        print(f"\n🔍 Checking {len(ACTIVE_POSITIONS)} active positions...")
                    monitor_active_positions(trader)
                    # HA reversal check — runs every time monitor runs
                    check_ha_reversal_alerts(access_token, trader)
                    last_position_check = current_time
            
            # Sync positions with broker every 5 minutes
            if ENABLE_EXIT_MANAGEMENT and trader:
                time_since_sync = (current_time - last_position_sync).seconds
                if time_since_sync >= 300:  # 5 minutes
                    if DEBUG_MODE:
                        print(f"\n🔄 Syncing positions with broker...")
                    sync_positions_with_broker(trader)
                    last_position_sync = current_time
            
            # Print position summary every 2 minutes
            if ACTIVE_POSITIONS:
                time_since_summary = (current_time - last_summary_print).seconds
                if time_since_summary >= 120:  # 2 minutes
                    print_position_summary()
                    last_summary_print = current_time
            
            # Check if it's exit time
            if ENABLE_EXIT_MANAGEMENT and ENABLE_TIME_BASED_EXIT and is_exit_time():
                if ACTIVE_POSITIONS:
                    print(f"\n⏰ EXIT TIME REACHED - Closing all positions")
                    exit_all_positions(trader, "TIME_BASED_EXIT")
                    break
                
            if not is_market_open():
                # Before closing, exit any remaining positions
                if ACTIVE_POSITIONS and trader:
                    print(f"\n⏰ Market closing - Exiting remaining positions")
                    exit_all_positions(trader, "MARKET_CLOSE")
                
                print(f"💤 Market closed. Waiting... ({datetime.now().strftime('%H:%M:%S')})", flush=True)
                time.sleep(60)
                continue
                
            # ── Fetch live prices + feed candle builder ALWAYS (even during stabilization) ──
            # This ensures real-time 5min candles accumulate from market open,
            # so fast trading has enough bars by the time stabilization ends.
            try:
                live_data = get_live_prices_batch(access_token, keys)
                if live_data:
                    for _rt_key, _rt_live in live_data.items():
                        _rt_symbol = ISIN_TO_SYMBOL.get(_rt_key, '')
                        if _rt_symbol:
                            update_realtime_candle(
                                _rt_symbol,
                                ltp=_rt_live.get('ltp'),
                                volume=_rt_live.get('volume', 0)
                            )
            except Exception as _candle_err:
                if DEBUG_MODE:
                    print(f"⚠️ Candle builder error: {_candle_err}")
                live_data = None

            if not is_market_stabilized():
                print(f"⏳ Market stabilizing... ({datetime.now().strftime('%H:%M:%S')})", flush=True)
                time.sleep(30)
                continue
                
            try:
                # Re-use live_data fetched above (avoids duplicate API call)
                if not live_data:
                    live_data = get_live_prices_batch(access_token, keys)
                
                if not live_data:
                    print("⚡ No data received", flush=True)
                    time.sleep(10)
                    continue
                
                # Check ORB time and process if needed
                if ENABLE_ORB_STRATEGY:
                    check_orb_time_and_process(access_token, live_data)
                    monitor_orb_breakouts(live_data, trader)
                
                # Update FII/DII if needed
                if ENABLE_FII_DII_FILTER:
                    update_fii_dii_if_needed()
                
                # Show pending confirmations
                if LAST_BREAKOUT_STATE and DEBUG_MODE:
                    print(f"\n📊 Pending R3/S3 confirmations: {len(LAST_BREAKOUT_STATE)}")

                # R3/S3 Breakout/Breakdown Logic (only if not stopped)
                if not TRADING_STOPPED:
                    for key, live in live_data.items():
                        if key in R3_LEVELS:
                            # R3 Breakout
                            breakout = check_breakout(key, live)
                            if breakout:
                                send_alert(breakout, trader)
                            
                            # S3 Breakdown
                            breakdown = check_breakdown(key, live)
                            if breakdown:
                                send_alert(breakdown, trader)
                        
            except Exception as e:
                print(f"❌ Monitor loop error: {e}")
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n⚡ Keyboard interrupt detected...")
        
        # Exit all open positions before stopping
        if ENABLE_EXIT_MANAGEMENT and trader and ACTIVE_POSITIONS:
            print(f"\n🚨 Exiting all open positions...")
            exit_all_positions(trader, "MANUAL_STOP")
        
        # Save final cache stats
        if ENABLE_CANDLE_CACHE:
            print("\n💾 Saving cache statistics...")
            save_cache_stats()
        
        print_final_stats()

# ========== MAIN EXECUTION ==========
def run_trading_bot(access_token):
    """Main execution function for trading bot"""
    banner()
    
    if not verify_token(access_token):
        print("❌ Cannot proceed without valid token")
        return
    
    # Get F&O stocks
    keys, symbols = get_all_fno_equities(access_token)
    if not keys:
        print("❌ No F&O stocks found")
        return
    
    # Initialize levels with retry
    max_init_attempts = 3
    init_success = False

    for attempt in range(max_init_attempts):
        print(f"\n📊 Initialization attempt {attempt + 1}/{max_init_attempts}")
        init_success = initialize_r3_levels(access_token, keys, symbols)

        if init_success:
            break
        elif attempt < max_init_attempts - 1:
            print(f"⏳ Waiting 10 seconds before retry {attempt + 2}...")
            time.sleep(10)
            reset_initialization()

    if not init_success:
        print("❌ Failed to initialize levels after multiple attempts")
        return
    
    # Create CSV files if needed
    csv_files_config = [
        (ALERT_CSV_FILE, ['Timestamp', 'Symbol', 'Type', 'Price', 'Volume_Ratio', 'Day_Gain', 'Strategy', 'Klinger_Status']),
        (EXIT_CSV_FILE, ['Exit_Time', 'Symbol', 'Strategy', 'Entry_Price', 'Exit_Price', 'Quantity', 'PnL', 'PnL_Percent', 'Exit_Reason', 'Order_ID']),
        (POSITION_LOG_FILE, ['Timestamp', 'Symbol', 'Strategy', 'Action', 'Price', 'Quantity', 'PnL', 'Details']),
    ]
    
    if ENABLE_ORB_STRATEGY:
        csv_files_config.extend([
            (ORB_SIGNALS_FILE, ['Timestamp', 'Symbol', 'Signal_Type', 'Direction', 
                               'Breakout_Level', 'Stop_Level', 'Target_Level', 
                               'Body_Percent', 'Risk_Reward', 'FII_DII_Signal', 'Confidence']),
            (ORB_TRADES_FILE, ['Timestamp', 'Symbol', 'Action', 'Direction', 'Price', 
                              'Stop_Loss', 'Target', 'Volume_Ratio', 'Confidence', 'FII_DII_Signal'])
        ])
    
    for csv_file, headers in csv_files_config:
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    # Session log
    with open(ALERT_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"SESSION START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stocks: {len(R3_LEVELS)}\n")
        f.write(f"Strategies: PullbackCE={'ON' if ENABLE_AUTO_TRADING and ENABLE_PULLBACK_CE_STRATEGY else 'OFF'} | ")
        f.write(f"S3={'ON' if ENABLE_AUTO_TRADING else 'OFF'} | ")
        f.write(f"ORB={'ON' if ENABLE_ORB_STRATEGY else 'OFF'}\n")
        f.write(f"🔥 Klinger Filter: {'ENABLED' if ENABLE_KLINGER_FILTER else 'DISABLED'}")
        if ENABLE_KLINGER_FILTER:
            f.write(f" (Fast={KLINGER_FAST}, Slow={KLINGER_SLOW}, Signal={KLINGER_SIGNAL})")
            f.write(f" Adaptive={'ON' if ADAPTIVE_KLINGER_LOOKBACK else 'OFF'}")
            if KLINGER_PAPER_MODE:
                f.write(f" - PAPER MODE")
        f.write(f"\n💾 Candle Cache: {'ENABLED' if ENABLE_CANDLE_CACHE else 'DISABLED'}")
        if ENABLE_CANDLE_CACHE:
            f.write(f" (Min candles: {MIN_CANDLES_FOR_KLINGER})")
        f.write(f"\n🚨 Exit Management: {'ENABLED' if ENABLE_EXIT_MANAGEMENT else 'DISABLED'}")
        if ENABLE_EXIT_MANAGEMENT:
            f.write(f" (Max Loss: ₹{MAX_DAILY_LOSS:,}, Max Profit: ₹{MAX_DAILY_PROFIT:,})")
        f.write(f"\n{'='*100}\n\n")
    
    print(f"\n✅ SYSTEM READY! Monitoring {len(R3_LEVELS)} high-volume F&O stocks")
    print(f"📊 Strategies: Pullback CE + S3 Breakdown{' + ORB' if ENABLE_ORB_STRATEGY else ''}")
    if ENABLE_KLINGER_FILTER:
        print(f"🔥 Klinger Oscillator: ACTIVE")
        print(f"   • Adaptive Mode: {'ENABLED' if ADAPTIVE_KLINGER_LOOKBACK else 'DISABLED'}")
        if KLINGER_PAPER_MODE:
            print(f"   • ✗ PAPER MODE: Logging only, not filtering")
    if ENABLE_CANDLE_CACHE:
        print(f"💾 Candle Cache: ACTIVE")
        print(f"   • Directory: {CACHE_DIRECTORY}")
        print(f"   • Min Candles: {MIN_CANDLES_FOR_KLINGER}")
    if ENABLE_ORB_STRATEGY:
        print(f"📊 ORB Strategy: ACTIVE")
        print(f"   • Timeframe: {ORB_TIMEFRAME_MINUTES} minutes")
        print(f"   • FII/DII Filter: {'ON' if ORB_ENABLE_FII_DII_FILTER else 'OFF'}")
    if ENABLE_FII_DII_FILTER:
        print(f"🔍 FII/DII Filter: ACTIVE")
        if FII_DII_STRONG_BUY:
            print(f"   • Strong Buy: {len(FII_DII_STRONG_BUY)} stocks")
        if FII_DII_STRONG_SELL:
            print(f"   • Strong Sell: {len(FII_DII_STRONG_SELL)} stocks")
        if ENABLE_FII_DII_TREND_FILTER:
            print(f"   • Multi-day Trend Filter: ACTIVE")
            with FII_DII_TREND_LOCK:
                n_acc = len(FII_DII_TREND_STRONG_ACCUMULATION)
                n_fbu = len(FII_DII_TREND_FII_BUY_DII_SELL)
                n_fse = len(FII_DII_TREND_FII_SELL_DII_BUY)
                n_unu = len(FII_DII_TREND_UNUSUAL_CHANGE)
            if n_acc:
                print(f"     Strong Accumulation: {n_acc} stocks")
            if n_fbu:
                print(f"     FII Buy / DII Sell : {n_fbu} stocks")
            if n_fse:
                print(f"     FII Sell / DII Buy : {n_fse} stocks")
            if n_unu:
                print(f"     Unusual Reversal   : {n_unu} stocks")
    if ENABLE_EXIT_MANAGEMENT:
        print(f"🚨 Exit Management: ACTIVE")
        print(f"   • Max Daily Loss: ₹{MAX_DAILY_LOSS:,}")
        print(f"   • Max Daily Profit: ₹{MAX_DAILY_PROFIT:,}")
        print(f"   • Trailing Stop: {TRAILING_STOP_PERCENTAGE}% (activates at {TRAILING_STOP_ACTIVATION}% profit)")
    print("🚀 Press Ctrl+C to stop\n")
    
    enhanced_monitor(access_token, keys, symbols)


def main():
    """Main entry point - get token and run trading bot"""
    print("="*120)
    print("UPSTOX AUTO-TRADING BOT WITH INTELLIGENT CACHING & ADVANCED STRATEGIES")
    print("="*120)
    print()
    
    # Step 1: Get Upstox access token (smart fallback)
    token = get_upstox_token()
    
    if not token:
        print("\n" + "="*60)
        print("❌ CRITICAL ERROR: Failed to get valid Upstox token")
        print("="*60)
        print("\nPlease check:")
        print("1. Your hardcoded token is correct and not expired")
        print("2. Your login credentials are correct")
        print("3. Your internet connection is stable")
        print("4. Upstox services are available")
        print("\nExiting...")
        return
    
    print("\n" + "="*60)
    print("✅ TOKEN ACQUIRED SUCCESSFULLY")
    print("="*60)
    print(f"Token: {token[:30]}...{token[-20:]}")
    print()
    
    # Step 2: Run trading bot with the token
    run_trading_bot(token)

if __name__ == "__main__":
    main()
