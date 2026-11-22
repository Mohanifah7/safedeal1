
from __future__ import annotations

from datetime import datetime, timezone, date
from io import StringIO

import ccxt
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crypto AutoHedge Lab",
    page_icon="📈",
    layout="wide",
)

STARTING_CASH = 10_000.00

NORMAL_RISK = 0.02
RECOVERY_RISK = 0.005
DAILY_RECOVERY_TRIGGER = 0.04

STOP_ATR_MULTIPLIER = 1.5
TARGET_ATR_MULTIPLIER = 3.0
TRAILING_ATR_MULTIPLIER = 1.5

MAX_POSITION_ALLOCATION = 0.25
MAX_OPEN_POSITIONS = 3
PAPER_FEE_RATE = 0.001

DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]

EXCHANGES = ["okx", "kucoin", "kraken"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


# ============================================================
# 2. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "cash": STARTING_CASH,
    "positions": {},
    "position_meta": {},
    "trades": [],
    "processed_candles": {},
    "auto_enabled": True,
    "daily_date": None,
    "daily_start_equity": None,
    "recovery_mode": False,
    "equity_history": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 3. EXCHANGE CONNECTION
# ============================================================

@st.cache_resource
def get_exchange(name: str):
    exchange_class = getattr(ccxt, name)

    return exchange_class({
        "enableRateLimit": True,
        "timeout": 20000,
    })


# ============================================================
# 4. MARKET DATA
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_scanner_data(
    symbols: tuple[str, ...],
    timeframe: str,
    limit: int = 200,
):
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
                    symbol_errors.append(
                        f"{exchange_name}: pair unavailable"
                    )
                    continue

                candles = client.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                )

                ticker = client.fetch_ticker(symbol)

                if not candles:
                    continue

                frame = pd.DataFrame(
                    candles,
                    columns=[
                        "Timestamp",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ],
                )

                frame["Timestamp"] = pd.to_datetime(
                    frame["Timestamp"],
                    unit="ms",
                    utc=True,
                )

                for column in [
                    "Open", "High", "Low", "Close", "Volume"
                ]:
                    frame[column] = pd.to_numeric(
                        frame[column],
                        errors="coerce",
                    )

                frame = frame.dropna(
                    subset=["Open", "High", "Low", "Close"]
                ).reset_index(drop=True)

                if len(frame) < 55:
                    continue

                # EMA indicators
                frame["EMA20"] = frame["Close"].ewm(
                    span=20,
                    adjust=False,
                ).mean()

                frame["EMA50"] = frame["Close"].ewm(
                    span=50,
                    adjust=False,
                ).mean()

                # RSI 14
                delta = frame["Close"].diff()

                gains = delta.clip(lower=0)
                losses = -delta.clip(upper=0)

                avg_gain = gains.ewm(
                    alpha=1 / 14,
                    adjust=False,
                    min_periods=14,
                ).mean()

                avg_loss = losses.ewm(
                    alpha=1 / 14,
                    adjust=False,
                    min_periods=14,
                ).mean()

                rs = avg_gain / avg_loss.where(avg_loss != 0)

                frame["RSI14"] = 100 - (
                    100 / (1 + rs)
                )

                frame.loc[
                    (avg_loss == 0) & (avg_gain > 0),
                    "RSI14",
                ] = 100

                frame.loc[
                    (avg_loss == 0) & (avg_gain == 0),
                    "RSI14",
                ] = 50

                # ATR 14
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
                    alpha=1 / 14,
                    adjust=False,
                    min_periods=14,
                ).mean()

                frame["VolumeMA20"] = frame["Volume"].rolling(
                    20,
                    min_periods=20,
                ).mean()

                price = ticker.get("last")

                if price is None:
                    price = frame["Close"].iloc[-1]

                price = float(price)

                if price <= 0:
                    raise ValueError("Invalid ticker price")

                output[symbol] = {
                    "frame": frame,
                    "price": price,
                    "change": float(
                        ticker.get("percentage") or 0
                    ),
                    "volume": float(
                        ticker.get("quoteVolume") or 0
                    ),
                    "exchange": exchange_name.upper(),
                }

                found = True
                break

            except Exception as exc:
                symbol_errors.append(
                    f"{exchange_name}: {str(exc)[:140]}"
                )

        if not found:
            errors[symbol] = " | ".join(symbol_errors)

    return output, errors


# ============================================================
# 5. INDICATORS AND SIGNALS
# ============================================================

def get_closed_candle(frame):
    if len(frame) < 2:
        return None

    return frame.iloc[-2]


def get_entry_signal(frame):
    candle = get_closed_candle(frame)

    if candle is None:
        return False

    required = [
        "Close", "EMA20", "EMA50",
        "RSI14", "ATR14", "Volume", "VolumeMA20",
    ]

    if candle[required].isna().any():
        return False

    close = float(candle["Close"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    rsi = float(candle["RSI14"])
    volume = float(candle["Volume"])
    average_volume = float(candle["VolumeMA20"])

    trend_ok = close > ema20 > ema50
    momentum_ok = 45 <= rsi <= 72
    volume_ok = (
        average_volume > 0
        and volume >= average_volume * 0.80
    )

    return trend_ok and momentum_ok and volume_ok


def get_signal_label(frame):
    candle = get_closed_candle(frame)

    if candle is None:
        return "INSUFFICIENT DATA"

    if candle[
        ["Close", "EMA20", "EMA50", "RSI14", "ATR14"]
    ].isna().any():
        return "INSUFFICIENT DATA"

    close = float(candle["Close"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    rsi = float(candle["RSI14"])

    if close > ema20 > ema50 and 45 <= rsi <= 72:
        return "BULLISH WATCH"

    if close < ema20 < ema50:
        return "BEARISH WATCH"

    if rsi > 72:
        return "OVERBOUGHT CAUTION"

    if rsi < 30:
        return "OVERSOLD CAUTION"

    return "NEUTRAL"


def confirmed_bearish_reversal(frame):
    """
    Require two completed candles to confirm deterioration.
    A single temporary RSI dip is not an exit by itself.
    """

    if len(frame) < 4:
        return False

    candles = frame.iloc[-3:-1]

    for _, candle in candles.iterrows():
        required = ["Close", "EMA20", "EMA50"]

        if candle[required].isna().any():
            return False

        if not (
            float(candle["Close"])
            < float(candle["EMA20"])
            < float(candle["EMA50"])
        ):
            return False

    return True


# ============================================================
# 6. ACCOUNT EQUITY
# ============================================================

def calculate_equity(market_data):
    position_value = 0.0

    for symbol, quantity in st.session_state.positions.items():
        data = market_data.get(symbol)
        meta = st.session_state.position_meta.get(symbol, {})

        if data:
            price = data["price"]
        else:
            price = meta.get("entry_price", 0.0)

        position_value += float(quantity) * float(price)

    return float(st.session_state.cash) + position_value


def update_daily_risk(market_data):
    equity = calculate_equity(market_data)

    today = datetime.now(timezone.utc).date().isoformat()

    if st.session_state.daily_date != today:

        st.session_state.daily_date = today
        st.session_state.daily_start_equity = equity
        st.session_state.recovery_mode = False

    if st.session_state.daily_start_equity is None:
        st.session_state.daily_start_equity = equity

    baseline = float(st.session_state.daily_start_equity)

    if baseline <= 0:
        return equity, 0.0, NORMAL_RISK

    daily_change = equity - baseline
    daily_pct = daily_change / baseline

    if daily_pct <= -DAILY_RECOVERY_TRIGGER:
        st.session_state.recovery_mode = True

    # Recovery mode ends only when daily starting equity is recovered.
    if st.session_state.recovery_mode and equity >= baseline:
        st.session_state.recovery_mode = False

    risk = (
        RECOVERY_RISK
        if st.session_state.recovery_mode
        else NORMAL_RISK
    )

    return equity, daily_pct, risk


# ============================================================
# 7. TRADE RECORDS
# ============================================================

def record_trade(
    symbol,
    side,
    quantity,
    price,
    fee,
    reason,
    meta=None,
    net_pnl=None,
):
    meta = meta or {}

    entry_time = meta.get("entry_time")

    holding_minutes = None

    if entry_time:
        try:
            started = datetime.fromisoformat(entry_time)
            now = datetime.now(timezone.utc)

            holding_minutes = max(
                0.0,
                (now - started).total_seconds() / 60,
            )
        except Exception:
            pass

    st.session_state.trades.insert(0, {
        "Time (UTC)": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "Symbol": symbol,
        "Side": side,
        "Quantity": quantity,
        "Price (USDT)": price,
        "Value (USDT)": quantity * price,
        "Fee (USDT)": fee,
        "Net P/L (USDT)": net_pnl,
        "Holding Minutes": holding_minutes,
        "MFE (%)": meta.get("mfe_pct"),
        "MAE (%)": meta.get("mae_pct"),
        "Entry RSI": meta.get("entry_rsi"),
        "Exit Reason": reason,
        "Mode": "PAPER",
    })


# ============================================================
# 8. OPEN POSITION
# ============================================================

def open_paper_position(
    symbol,
    data,
    candle,
    risk_fraction,
):
    price = float(data["price"])
    atr = float(candle["ATR14"])

    if price <= 0 or atr <= 0:
        return False

    stop_distance = atr * STOP_ATR_MULTIPLIER

    if stop_distance >= price:
        return False

    equity = calculate_equity({symbol: data})

    risk_budget = equity * risk_fraction

    risk_quantity = risk_budget / stop_distance

    max_allocation = equity * MAX_POSITION_ALLOCATION
    allocation_quantity = max_allocation / price

    cash_quantity = (
        st.session_state.cash
        / (price * (1 + PAPER_FEE_RATE))
    )

    quantity = min(
        risk_quantity,
        allocation_quantity,
        cash_quantity,
    )

    notional = quantity * price
    fee = notional * PAPER_FEE_RATE

    if quantity <= 0 or notional < 10:
        return False

    if notional + fee > st.session_state.cash:
        return False

    stop_price = price - stop_distance

    target_price = (
        price + atr * TARGET_ATR_MULTIPLIER
    )

    st.session_state.cash -= notional + fee

    positions = dict(st.session_state.positions)
    positions[symbol] = quantity
    st.session_state.positions = positions

    meta = dict(st.session_state.position_meta)

    meta[symbol] = {
        "entry_price": price,
        "stop_price": stop_price,
        "initial_stop_price": stop_price,
        "target_price": target_price,
        "entry_atr": atr,
        "risk_amount": risk_budget,
        "entry_fee": fee,
        "entry_time": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "highest_price": price,
        "lowest_price": price,
        "mfe_pct": 0.0,
        "mae_pct": 0.0,
        "trailing_active": False,
        "target_reached": False,
        "entry_rsi": float(candle["RSI14"]),
        "entry_ema20": float(candle["EMA20"]),
        "entry_ema50": float(candle["EMA50"]),
        "entry_signal": "Aggressive bullish trend",
    }

    st.session_state.position_meta = meta

    record_trade(
        symbol,
        "BUY",
        quantity,
        price,
        fee,
        "Automatic bullish entry",
        meta[symbol],
    )

    return True


# ============================================================
# 9. CLOSE POSITION
# ============================================================

def close_paper_position(symbol, price, reason):
    positions = dict(st.session_state.positions)

    quantity = float(positions.get(symbol, 0.0))

    if quantity <= 0 or price <= 0:
        return False

    meta = dict(
        st.session_state.position_meta.get(symbol, {})
    )

    entry = float(meta.get("entry_price", price))

    proceeds = quantity * price
    exit_fee = proceeds * PAPER_FEE_RATE

    entry_fee = float(meta.get("entry_fee", 0.0))

    gross_pnl = (price - entry) * quantity

    net_pnl = gross_pnl - entry_fee - exit_fee

    st.session_state.cash += proceeds - exit_fee

    positions.pop(symbol, None)

    position_meta = dict(st.session_state.position_meta)
    position_meta.pop(symbol, None)

    st.session_state.positions = positions
    st.session_state.position_meta = position_meta

    record_trade(
        symbol,
        "SELL",
        quantity,
        price,
        exit_fee,
        reason,
        meta,
        net_pnl,
    )

    return True


# ============================================================
# 10. ADAPTIVE POSITION MANAGEMENT
# ============================================================

def manage_position(symbol, data, candle):
    quantity = float(
        st.session_state.positions.get(symbol, 0.0)
    )

    if quantity <= 0:
        return None

    meta = dict(
        st.session_state.position_meta.get(symbol, {})
    )

    # Old positions without metadata must be recovered manually.
    required = [
        "entry_price",
        "stop_price",
        "target_price",
        "entry_atr",
    ]

    if any(meta.get(key) is None for key in required):
        return "Position requires metadata recovery"

    price = float(data["price"])
    entry = float(meta["entry_price"])
    entry_atr = float(meta["entry_atr"])

    if entry <= 0 or entry_atr <= 0:
        return "Invalid position metadata"

    # Track maximum favourable and adverse price.
    highest = max(
        float(meta.get("highest_price", entry)),
        price,
    )

    lowest = min(
        float(meta.get("lowest_price", entry)),
        price,
    )

    meta["highest_price"] = highest
    meta["lowest_price"] = lowest

    meta["mfe_pct"] = (
        (highest - entry) / entry
    ) * 100

    meta["mae_pct"] = (
        (lowest - entry) / entry
    ) * 100

    target = float(meta["target_price"])

    # Activate trailing protection after the initial
    # 3 ATR profit objective is reached.
    if highest >= target:

        meta["target_reached"] = True
        meta["trailing_active"] = True

    # Once activated, trail behind the highest observed price.
    if meta.get("trailing_active"):

        trailing_stop = (
            highest - entry_atr * TRAILING_ATR_MULTIPLIER
        )

        meta["stop_price"] = max(
            float(meta["stop_price"]),
            trailing_stop,
        )

    # Save updated stop and excursion values.
    updated = dict(st.session_state.position_meta)
    updated[symbol] = meta
    st.session_state.position_meta = updated

    stop_price = float(meta["stop_price"])

    # Hard protective stop always takes priority.
    if price <= stop_price:

        reason = (
            "Trailing stop triggered"
            if meta.get("trailing_active")
            else "Initial ATR stop-loss triggered"
        )

        if close_paper_position(symbol, price, reason):
            return f"SELL {symbol}: {reason}"

    # Trend reversal must be confirmed on two closed candles.
    if confirmed_bearish_reversal(data["frame"]):

        if close_paper_position(
            symbol,
            price,
            "Confirmed bearish EMA reversal",
        ):
            return (
                f"SELL {symbol}: confirmed bearish trend reversal"
            )

    # No forced exit merely because RSI temporarily falls below 40.
    return None


# ============================================================
# 11. AUTOMATIC TRADING ENGINE
# ============================================================

def process_automatic_trades(market_data, risk_fraction):
    messages = []

    for symbol, data in market_data.items():

        frame = data["frame"]
        candle = get_closed_candle(frame)

        if candle is None:
            continue

        candle_id = str(candle["Timestamp"])

        previous_id = (
            st.session_state.processed_candles.get(symbol)
        )

        already_processed = previous_id == candle_id

        quantity = float(
            st.session_state.positions.get(symbol, 0.0)
        )

        # Manage existing positions on every app rerun.
        if quantity > 0:

            result = manage_position(symbol, data, candle)

            if result:
                messages.append(result)

            # Do not re-enter on the same candle.
            st.session_state.processed_candles[
                symbol
            ] = candle_id

            continue

        if already_processed:
            continue

        st.session_state.processed_candles[
            symbol
        ] = candle_id

        if not get_entry_signal(frame):
            continue

        open_count = sum(
            1
            for qty in st.session_state.positions.values()
            if float(qty) > 0
        )

        if open_count >= MAX_OPEN_POSITIONS:
            messages.append(
                f"SKIP {symbol}: maximum open positions reached"
            )
            continue

        if open_paper_position(
            symbol,
            data,
            candle,
            risk_fraction,
        ):
            messages.append(
                f"BUY {symbol}: automatic paper entry "
                f"using {risk_fraction * 100:.2f}% risk"
            )

    return messages


# ============================================================
# 12. HEADER
# ============================================================

st.title("Crypto AutoHedge Lab")

st.caption(
    "Stage 3 · Adaptive holding · Recovery mode · "
    "Automatic paper execution · No real exchange orders"
)

st.divider()


# ============================================================
# 13. SCANNER SETTINGS
# ============================================================

st.header("Market Scanner")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_symbols = st.multiselect(
        "Assets",
        DEFAULT_SYMBOLS,
        default=DEFAULT_SYMBOLS,
    )

with col2:
    timeframe = st.selectbox(
        "Timeframe",
        TIMEFRAMES,
        index=1,
    )

with col3:
    st.write(" ")
    refresh = st.button(
        "Refresh market data",
        use_container_width=True,
    )

if refresh:
    fetch_scanner_data.clear()


# ============================================================
# 14. AUTOMATION CONTROL
# ============================================================

st.subheader("Automatic Paper Execution")

st.session_state.auto_enabled = st.toggle(
    "Enable automatic paper trading",
    value=st.session_state.auto_enabled,
)

if st.session_state.auto_enabled:
    st.success("Automatic paper execution is enabled.")
else:
    st.warning(
        "Automatic execution is paused. "
        "Existing positions remain open."
    )

if not selected_symbols:
    st.warning("Select at least one asset.")
    st.stop()


# ============================================================
# 15. FETCH MARKET DATA
# ============================================================

with st.spinner("Loading public exchange market data..."):

    symbols_to_fetch = tuple(dict.fromkeys(
        list(selected_symbols)
        + list(st.session_state.positions.keys())
    ))

    market_data, market_errors = fetch_scanner_data(
        symbols_to_fetch,
        timeframe,
    )


# ============================================================
# 16. DAILY RISK AND RECOVERY MODE
# ============================================================

equity_before = calculate_equity(market_data)

equity, daily_pct, current_risk = update_daily_risk(
    market_data
)

st.divider()
st.header("Portfolio Risk Control")

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Daily Equity Change",
    f"{daily_pct * 100:+.2f}%",
)

r2.metric(
    "Current Risk per Trade",
    f"{current_risk * 100:.2f}%",
)

r3.metric(
    "Daily Loss Trigger",
    "-4.00%",
)

r4.metric(
    "Open Positions",
    sum(
        1
        for qty in st.session_state.positions.values()
        if float(qty) > 0
    ),
)

if st.session_state.recovery_mode:

    st.warning(
        "RECOVERY MODE ACTIVE: Daily drawdown reached 4%. "
        "New qualifying paper trades use 0.5% risk per trade. "
        "Existing positions remain managed by their stops and "
        "confirmed trend exits. Trading risk returns to 2% only "
        "when daily starting equity is recovered."
    )

else:

    st.success(
        "NORMAL MODE: Standard 2% virtual risk per trade."
    )

if timeframe != "15m":

    st.warning(
        "Automatic entries are configured for 15m candles only. "
        "Switch to 15m to enable the automatic trading engine."
    )


# ============================================================
# 17. PROCESS AUTOMATIC TRADES
# ============================================================

execution_messages = []

if (
    st.session_state.auto_enabled
    and timeframe == "15m"
):

    execution_messages = process_automatic_trades(
        market_data,
        current_risk,
    )

for message in execution_messages:

    if message.startswith("SELL"):
        st.warning(message)

    elif message.startswith("SKIP"):
        st.info(message)

    else:
        st.success(message)


# Recalculate after any paper trades.
equity = calculate_equity(market_data)

equity, daily_pct, current_risk = update_daily_risk(
    market_data
)


# ============================================================
# 18. MARKET OVERVIEW
# ============================================================

st.divider()
st.header("Market Overview")

if market_data:

    rows = []

    for symbol, data in market_data.items():

        candle = get_closed_candle(data["frame"])

        if candle is None:
            continue

        rows.append({
            "Symbol": symbol,
            "Price (USDT)": data["price"],
            "24h Change (%)": data["change"],
            "EMA20": candle["EMA20"],
            "EMA50": candle["EMA50"],
            "RSI14": candle["RSI14"],
            "ATR14": candle["ATR14"],
            "Signal": get_signal_label(data["frame"]),
            "Exchange": data["exchange"],
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

else:
    st.warning("No market data currently available.")

if market_errors:

    with st.expander("Market data errors"):

        for symbol, error in market_errors.items():
            st.write(f"**{symbol}:** {error}")


# ============================================================
# 19. ASSET ANALYSIS
# ============================================================

st.divider()
st.header("Asset Analysis")

available_symbols = list(market_data.keys())

if available_symbols:

    selected_asset = st.selectbox(
        "Select asset",
        available_symbols,
    )

    asset = market_data[selected_asset]
    frame = asset["frame"]
    candle = get_closed_candle(frame)

    if candle is not None:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current Price",
            f"${asset['price']:,.4f}",
        )

        c2.metric(
            "EMA20",
            f"${candle['EMA20']:,.4f}",
        )

        c3.metric(
            "EMA50",
            f"${candle['EMA50']:,.4f}",
        )

        c4.metric(
            "RSI14",
            f"{candle['RSI14']:.2f}",
        )

        st.write(
            f"**Signal:** {get_signal_label(frame)}"
        )

        st.line_chart(
            frame.set_index("Timestamp")[
                ["Close", "EMA20", "EMA50"]
            ]
        )


# ============================================================
# 20. ACCOUNT OVERVIEW
# ============================================================

st.divider()
st.header("Account Overview")

equity = calculate_equity(market_data)

a1, a2, a3, a4 = st.columns(4)

a1.metric(
    "Starting Capital",
    f"${STARTING_CASH:,.2f}",
)

a2.metric(
    "Available Cash",
    f"${st.session_state.cash:,.2f}",
)

a3.metric(
    "Estimated Equity",
    f"${equity:,.2f}",
)

a4.metric(
    "Closed Paper Trades",
    sum(
        1
        for trade in st.session_state.trades
        if trade.get("Side") == "SELL"
    ),
)


# ============================================================
# 21. RISK MONITOR AND POSITION RECOVERY
# ============================================================

st.divider()
st.header("Risk Monitor")

open_positions = {
    symbol: float(quantity)
    for symbol, quantity in st.session_state.positions.items()
    if float(quantity) > 0
}

if open_positions:

    st.caption(
        "P/L is estimated from public ticker prices and excludes "
        "future exit fees and slippage. Recovery calculations are "
        "session-based."
    )

    for symbol, quantity in open_positions.items():

        data = market_data.get(symbol)
        meta = st.session_state.position_meta.get(symbol, {})

        missing = any(
            meta.get(key) is None
            for key in [
                "entry_price",
                "stop_price",
                "target_price",
                "entry_atr",
            ]
        )

        if missing:

            st.warning(
                f"{symbol} needs its original paper-trade "
                "entry and risk details."
            )

            default_entry = (
                59244.10 if symbol == "BTC/USDT" else 0.0
            )

            previous_buys = [
                trade
                for trade in st.session_state.trades
                if trade.get("Symbol") == symbol
                and trade.get("Side") == "BUY"
            ]

            if previous_buys and symbol != "BTC/USDT":
                default_entry = float(
                    previous_buys[0].get("Price (USDT)", 0)
                )

            with st.form(key=f"recover_{symbol}"):

                recovery_entry = st.number_input(
                    "Original entry price (USDT)",
                    min_value=0.0,
                    value=float(default_entry),
                    step=0.01,
                    format="%.8f",
                )

                submitted = st.form_submit_button(
                    "Save entry and calculate risk levels",
                    use_container_width=True,
                )

                if submitted:

                    if recovery_entry <= 0:
                        st.error("Enter a valid entry price.")

                    elif data is None:
                        st.error("Market data is unavailable.")

                    else:

                        recovery_candle = get_closed_candle(
                            data["frame"]
                        )

                        atr = (
                            float(recovery_candle["ATR14"])
                            if recovery_candle is not None
                            and pd.notna(recovery_candle["ATR14"])
                            else 0.0
                        )

                        if atr <= 0:
                            st.error("ATR14 is unavailable.")

                        else:

                            entry = float(recovery_entry)
                            stop = (
                                entry - atr * STOP_ATR_MULTIPLIER
                            )
                            target = (
                                entry + atr * TARGET_ATR_MULTIPLIER
                            )

                            if stop <= 0:
                                st.error("Invalid calculated stop.")

                            else:

                                recovered = dict(
                                    st.session_state.position_meta
                                )

                                recovered[symbol] = {
                                    "entry_price": entry,
                                    "stop_price": stop,
                                    "initial_stop_price": stop,
                                    "target_price": target,
                                    "entry_atr": atr,
                                    "risk_amount": (
                                        quantity * (entry - stop)
                                    ),
                                    "entry_fee": 0.0,
                                    "entry_time": datetime.now(
                                        timezone.utc
                                    ).isoformat(timespec="seconds"),
                                    "highest_price": max(
                                        entry, data["price"]
                                    ),
                                    "lowest_price": min(
                                        entry, data["price"]
                                    ),
                                    "mfe_pct": 0.0,
                                    "mae_pct": 0.0,
                                    "trailing_active": False,
                                    "target_reached": False,
                                    "entry_rsi": None,
                                    "entry_ema20": None,
                                    "entry_ema50": None,
                                    "entry_signal": "Recovered legacy position",
                                    "recovered": True,
                                }

                                st.session_state.position_meta = recovered

                                st.success(
                                    f"{symbol} metadata saved."
                                )

                                st.rerun()

        # Reload metadata after any recovery.
        meta = st.session_state.position_meta.get(symbol, {})

        entry = meta.get("entry_price")
        stop = meta.get("stop_price")
        target = meta.get("target_price")

        entry = float(entry) if entry is not None else None
        stop = float(stop) if stop is not None else None
        target = float(target) if target is not None else None

        current = (
            float(data["price"])
            if data is not None
            else None
        )

        pnl = None
        pnl_pct = None

        if entry is not None and current is not None and entry > 0:

            pnl = (current - entry) * quantity
            pnl_pct = ((current - entry) / entry) * 100

        with st.container(border=True):

            st.subheader(f"🪙 {symbol}")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Entry Price",
                f"${entry:,.6f}" if entry is not None else "Unavailable",
            )

            c2.metric(
                "Current Price",
                f"${current:,.6f}" if current is not None else "Unavailable",
            )

            c3.metric(
                "Quantity",
                f"{quantity:.8f}",
            )

            c4, c5, c6 = st.columns(3)

            c4.metric(
                "Current Stop Loss",
                f"${stop:,.6f}" if stop is not None else "Not set",
            )

            c5.metric(
                "Initial Take Profit",
                f"${target:,.6f}" if target is not None else "Not set",
            )

            if pnl is not None:

                c6.metric(
                    "Unrealized P/L",
                    f"${pnl:+,.4f}",
                    delta=f"{pnl_pct:+.2f}%",
                )

            else:
                c6.metric("Unrealized P/L", "Unavailable")

            if meta:

                trailing = meta.get("trailing_active", False)
                target_reached = meta.get("target_reached", False)

                st.write(
                    f"**Trailing protection:** "
                    f"{'Active' if trailing else 'Not active'}"
                )

                st.write(
                    f"**Initial target reached:** "
                    f"{'Yes' if target_reached else 'No'}"
                )

                entry_time = meta.get("entry_time")

                if entry_time:
                    try:
                        started = datetime.fromisoformat(entry_time)
                        minutes = max(
                            0,
                            (
                                datetime.now(timezone.utc) - started
                            ).total_seconds() / 60,
                        )

                        st.write(
                            f"**Holding duration:** "
                            f"{minutes / 60:.1f} hours"
                        )
                    except Exception:
                        pass

                if meta.get("mfe_pct") is not None:
                    st.write(
                        f"**Maximum favourable movement:** "
                        f"{meta['mfe_pct']:.2f}%"
                    )

                if meta.get("mae_pct") is not None:
                    st.write(
                        f"**Maximum adverse movement:** "
                        f"{meta['mae_pct']:.2f}%"
                    )

                if meta.get("recovered"):
                    st.caption(
                        "Recovered legacy position. "
                        "Original entry supplied manually."
                    )

else:

    st.info("No open virtual positions.")


# ============================================================
# 22. PERFORMANCE ANALYSIS
# ============================================================

st.divider()
st.header("Performance Analysis")

closed_trades = [
    trade
    for trade in st.session_state.trades
    if trade.get("Side") == "SELL"
    and trade.get("Net P/L (USDT)") is not None
]

if closed_trades:

    results = [
        float(trade["Net P/L (USDT)"])
        for trade in closed_trades
    ]

    wins = [value for value in results if value > 0]
    losses = [value for value in results if value < 0]

    win_rate = len(wins) / len(results) * 100

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    total_realized = sum(results)

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Realized P/L",
        f"${total_realized:+,.2f}",
    )

    p2.metric(
        "Win Rate",
        f"{win_rate:.1f}%",
    )

    p3.metric(
        "Profit Factor",
        f"{profit_factor:.2f}"
        if profit_factor is not None else "N/A",
    )

    p4.metric(
        "Closed Trades",
        len(results),
    )

    st.caption(
        "Performance statistics use recorded closed paper trades. "
        "Recovered legacy positions may have estimated historical fees."
    )

else:

    st.info(
        "Performance statistics will appear after your first "
        "completed paper trade."
    )


# ============================================================
# 23. TRADE HISTORY AND EXPORT
# ============================================================

st.divider()
st.header("Paper Trade History")

if st.session_state.trades:

    trade_frame = pd.DataFrame(
        st.session_state.trades
    )

    st.dataframe(
        trade_frame,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = trade_frame.to_csv(index=False)

    st.download_button(
        "Download paper trade history (CSV)",
        data=csv_data,
        file_name="crypto_autohedge_paper_trades.csv",
        mime="text/csv",
    )

else:

    st.caption("No paper trades recorded yet.")


# ============================================================
# 24. EQUITY TRACKING
# ============================================================

st.divider()
st.header("Equity Tracking")

history = list(st.session_state.equity_history)

snapshot_time = datetime.now(timezone.utc).isoformat(
    timespec="seconds"
)

# Keep one snapshot per minute to avoid duplicate rerun entries.
if (
    not history
    or history[-1].get("Time") is None
    or (
        datetime.now(timezone.utc)
        - datetime.fromisoformat(history[-1]["Time"])
    ).total_seconds() >= 60
):

    history.append({
        "Time": snapshot_time,
        "Equity": equity,
    })

    history = history[-2000:]

    st.session_state.equity_history = history

if history:

    equity_frame = pd.DataFrame(history)

    st.line_chart(
        equity_frame.set_index("Time")["Equity"]
    )

    st.caption(
        "Equity history is recorded during active app sessions. "
        "It is not a permanent database."
    )


# ============================================================
# 25. STRATEGY CONFIGURATION
# ============================================================

st.divider()
st.header("Stage 3 Strategy Configuration")

settings = {
    "Strategy": "Aggressive adaptive",
    "Execution": "Automatic paper trading",
    "Normal risk": "2% per trade",
    "Daily recovery trigger": "-4% of daily starting equity",
    "Recovery risk": "0.5% per trade",
    "Recovery objective": "Recover daily starting equity",
    "Initial stop loss": "1.5 × ATR14",
    "Initial profit objective": "3 × ATR14",
    "Trailing stop": "1.5 × entry ATR below highest price",
    "Trend exit": "Two confirmed bearish completed candles",
    "Maximum positions": "3",
    "Maximum allocation": "25% per position",
    "Simulated fee": "0.1% per side",
    "Entry timeframe": "15m",
    "Trading direction": "Long only",
}

st.dataframe(
    pd.DataFrame(
        list(settings.items()),
        columns=["Setting", "Value"],
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 26. FOOTER
# ============================================================

st.caption(
    "Educational paper-trading simulator only. "
    "No real exchange orders or wallet access. "
    "Public exchange data may be delayed or unavailable. "
    "Automatic processing occurs only when the Streamlit app reruns. "
    "Stops cannot guarantee execution prices. "
    "Simulated fees are estimates; slippage and funding are excluded. "
    "Positions, daily risk state, and history are session-based "
    "and may reset when the session restarts. "
    "This system does not guarantee recovery or profitability."
)
