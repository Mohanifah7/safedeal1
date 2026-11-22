from __future__ import annotations

from datetime import datetime, timezone

import ccxt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto AutoHedge Lab", page_icon="📈", layout="wide")

STARTING_CASH = 10_000.00
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
EXCHANGES = ["okx", "kucoin", "kraken"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

st.title("Crypto AutoHedge Lab")
st.caption("Research workspace · Paper trading only · No real orders are sent")

if "cash" not in st.session_state:
    st.session_state.cash = STARTING_CASH
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "trades" not in st.session_state:
    st.session_state.trades = []

@st.cache_resource
def get_exchange(name: str):
    exchange_class = getattr(ccxt, name)
    return exchange_class({"enableRateLimit": True, "timeout": 20000})

@st.cache_data(ttl=60, show_spinner=False)
def fetch_scanner_data(symbols: tuple[str, ...], timeframe: str, limit: int = 200):
    """Try public exchanges in order; return OHLCV and ticker data per supported asset."""
    output = {}
    errors = {}

    for symbol in symbols:
        found = False
        symbol_errors = []
        for exchange_name in EXCHANGES:
            try:
                client = get_exchange(exchange_name)
                markets = client.load_markets()
                if symbol not in markets:
                    symbol_errors.append(f"{exchange_name}: pair unavailable")
                    continue

                candles = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                ticker = client.fetch_ticker(symbol)
                if not candles:
                    symbol_errors.append(f"{exchange_name}: no candles returned")
                    continue

                frame = pd.DataFrame(
                    candles,
                    columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
                )
                frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], unit="ms", utc=True)
                for column in ["Open", "High", "Low", "Close", "Volume"]:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")

                frame["EMA20"] = frame["Close"].ewm(span=20, adjust=False).mean()
                frame["EMA50"] = frame["Close"].ewm(span=50, adjust=False).mean()

                delta = frame["Close"].diff()
                gains = delta.clip(lower=0)
                losses = -delta.clip(upper=0)
                avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
                avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
                relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
                frame["RSI14"] = 100 - (100 / (1 + relative_strength))
                frame.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100
                frame.loc[(avg_loss == 0) & (avg_gain == 0), "RSI14"] = 50

                previous_close = frame["Close"].shift(1)
                true_range = pd.concat(
                    [
                        frame["High"] - frame["Low"],
                        (frame["High"] - previous_close).abs(),
                        (frame["Low"] - previous_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                frame["ATR14"] = true_range.ewm(
                    alpha=1 / 14, adjust=False, min_periods=14
                ).mean()

                output[symbol] = {
                    "frame": frame,
                    "price": float(ticker.get("last") or frame["Close"].iloc[-1]),
                    "change": float(ticker.get("percentage") or 0),
                    "volume": float(ticker.get("quoteVolume") or 0),
                    "exchange": exchange_name.upper(),
                }
                found = True
                break
