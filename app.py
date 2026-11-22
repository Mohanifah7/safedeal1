
from __future__ import annotations

from datetime import datetime, timezone

import ccxt
import pandas as pd
import streamlit as st

# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Crypto AutoHedge Lab",
    page_icon="📈",
    layout="wide",
)

STARTING_CASH = 10_000.00
RISK_PER_TRADE = 0.02
STOP_ATR_MULTIPLIER = 1.5
TARGET_ATR_MULTIPLIER = 3.0
MAX_POSITION_ALLOCATION = 0.25
PAPER_FEE_RATE = 0.001

DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]

EXCHANGES = ["okx", "kucoin", "kraken"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# ==================================================
# 2. SESSION STATE
# ==================================================

if "cash" not in st.session_state:
    st.session_state.cash = STARTING_CASH

if "positions" not in st.session_state:
    st.session_state.positions = {}

if "position_meta" not in st.session_state:
    st.session_state.position_meta = {}

if "trades" not in st.session_state:
    st.session_state.trades = []

if "processed_candles" not in st.session_state:
    st.session_state.processed_candles = {}

if "auto_enabled" not in st.session_state:
    st.session_state.auto_enabled = True

# ==================================================
# 3. EXCHANGE CONNECTION
# ==================================================

@st.cache_resource
def get_exchange(name: str):
    exchange_class = getattr(ccxt, name)

    return exchange_class({
        "enableRateLimit": True,
        "timeout": 20000,
    })


# ==================================================
# 4. MARKET DATA AND INDICATORS
# ==================================================

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
                    symbol_errors.append(
                        f"{exchange_name}: no candles returned"
                    )
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
                    symbol_errors.append(
                        f"{exchange_name}: insufficient candle history"
                    )
                    continue

                # EMA 20 and EMA 50
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

                relative_strength = (
                    avg_gain / avg_loss.where(avg_loss != 0)
                )

                frame["RSI14"] = (
                    100 - (100 / (1 + relative_strength))
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

                # Average volume
                frame["VolumeMA20"] = frame["Volume"].rolling(
                    window=20,
                    min_periods=20,
                ).mean()

                last_price = ticker.get("last")

                if last_price is None:
                    last_price = frame["Close"].iloc[-1]

                if float(last_price) <= 0:
                    raise ValueError("Invalid market price")

                output[symbol] = {
                    "frame": frame,
                    "price": float(last_price),
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
                    f"{exchange_name}: {str(exc)[:160]}"
                )
                continue

        if not found:
            errors[symbol] = " | ".join(symbol_errors)

    return output, errors


# ==================================================
# 5. SIGNAL ENGINE
# ==================================================

def get_closed_candle(frame):
    """
    Use the previous candle because the newest candle
    may still be forming.
    """
    if len(frame) < 2:
        return None

    return frame.iloc[-2]


def get_entry_signal(frame):
    candle = get_closed_candle(frame)

    if candle is None:
        return False

    required = [
        "Close",
        "EMA20",
        "EMA50",
        "RSI14",
        "ATR14",
        "Volume",
        "VolumeMA20",
    ]

    if candle[required].isna().any():
        return False

    close = float(candle["Close"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    rsi = float(candle["RSI14"])
    volume = float(candle["Volume"])
    average_volume = float(candle["VolumeMA20"])

    # Aggressive long-entry conditions
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

    if rsi > 72 or rsi < 30:
        return "CAUTION"

    return "NEUTRAL"


# ==================================================
# 6. ACCOUNT EQUITY
# ==================================================

def calculate_equity(market_data):
    position_value = 0.0

    for symbol, quantity in st.session_state.positions.items():
        data = market_data.get(symbol)

        if data:
            price = data["price"]
        else:
            meta = st.session_state.position_meta.get(symbol, {})
            price = meta.get("entry_price", 0.0)

        position_value += float(quantity) * float(price)

    return float(st.session_state.cash) + position_value


# ==================================================
# 7. PAPER TRADE EXECUTION
# ==================================================

def record_trade(
    symbol,
    side,
    quantity,
    price,
    fee,
    reason,
    risk_amount=0.0,
):
    value = quantity * price

    st.session_state.trades.insert(0, {
        "Time (UTC)": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "Symbol": symbol,
        "Side": side,
        "Quantity": quantity,
        "Price (USDT)": price,
        "Value (USDT)": value,
        "Fee (USDT)": fee,
        "Risk Budget (USDT)": risk_amount,
        "Reason": reason,
        "Mode": "PAPER",
    })


def open_paper_position(symbol, data, candle):
    price = float(data["price"])
    atr = float(candle["ATR14"])

    if atr <= 0 or price <= 0:
        return False

    stop_distance = atr * STOP_ATR_MULTIPLIER

    if stop_distance <= 0 or stop_distance >= price:
        return False

    equity = calculate_equity({symbol: data})
    risk_budget = equity * RISK_PER_TRADE

    # Position quantity based on 2% risk budget
    risk_quantity = risk_budget / stop_distance

    # Limit position size to 25% of account equity
    max_allocation = equity * MAX_POSITION_ALLOCATION
    allocation_quantity = max_allocation / price

    # Include simulated entry fee in available cash
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
    target_price = price + (
        atr * TARGET_ATR_MULTIPLIER
    )

    st.session_state.cash -= notional + fee

    positions = dict(st.session_state.positions)
    positions[symbol] = quantity
    st.session_state.positions = positions

    meta = dict(st.session_state.position_meta)

    meta[symbol] = {
        "entry_price": price,
        "stop_price": stop_price,
        "target_price": target_price,
        "entry_atr": atr,
        "risk_amount": risk_budget,
        "entry_time": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
    }

    st.session_state.position_meta = meta

    record_trade(
        symbol=symbol,
        side="BUY",
        quantity=quantity,
        price=price,
        fee=fee,
        reason="Automatic bullish entry",
        risk_amount=risk_budget,
    )

    return True


def close_paper_position(symbol, price, reason):
    positions = dict(st.session_state.positions)

    quantity = float(positions.get(symbol, 0.0))

    if quantity <= 0 or price <= 0:
        return False

    proceeds = quantity * price
    fee = proceeds * PAPER_FEE_RATE

    st.session_state.cash += proceeds - fee

    meta = dict(st.session_state.position_meta)
    meta.pop(symbol, None)

    positions.pop(symbol, None)

    st.session_state.positions = positions
    st.session_state.position_meta = meta

    record_trade(
        symbol=symbol,
        side="SELL",
        quantity=quantity,
        price=price,
        fee=fee,
        reason=reason,
    )

    return True


# ==================================================
# 8. AUTOMATIC ENTRY AND EXIT ENGINE
# ==================================================

def process_automatic_trades(market_data):
    messages = []

    for symbol, data in market_data.items():

        frame = data["frame"]
        current_price = float(data["price"])
        candle = get_closed_candle(frame)

        if candle is None:
            continue

        candle_id = str(candle["Timestamp"])

        previous_candle_id = (
            st.session_state.processed_candles.get(symbol)
        )

        already_processed = (
            previous_candle_id == candle_id
        )

        # ------------------------------------------
        # Check existing position exits
        # ------------------------------------------

        quantity = float(
            st.session_state.positions.get(symbol, 0.0)
        )

        meta = st.session_state.position_meta.get(symbol)

        if quantity > 0 and meta:

            stop_price = float(meta["stop_price"])
            target_price = float(meta["target_price"])

            exit_reason = None

            if current_price <= stop_price:
                exit_reason = "ATR stop-loss triggered"

            elif current_price >= target_price:
                exit_reason = "ATR take-profit triggered"

            else:
                close = float(candle["Close"])
                ema20 = float(candle["EMA20"])
                ema50 = float(candle["EMA50"])
                rsi = float(candle["RSI14"])

                if close < ema20 and ema20 < ema50:
                    exit_reason = "Bearish EMA trend exit"

                elif rsi < 40:
                    exit_reason = "Momentum exit: RSI below 40"

            if exit_reason:
                if close_paper_position(
                    symbol,
                    current_price,
                    exit_reason,
                ):
                    messages.append(
                        f"SELL {symbol}: {exit_reason}"
                    )

            # Do not re-enter on the same candle after an exit
            st.session_state.processed_candles[
                symbol
            ] = candle_id

            continue

        # ------------------------------------------
        # Prevent duplicate entries per candle
        # ------------------------------------------

        if already_processed:
            continue

        st.session_state.processed_candles[
            symbol
        ] = candle_id

        # ------------------------------------------
        # Automatic entry
        # ------------------------------------------

        if quantity == 0 and get_entry_signal(frame):

            if open_paper_position(symbol, data, candle):
                messages.append(
                    f"BUY {symbol}: automatic bullish entry"
                )

    return messages


# ==================================================
# 9. HEADER
# ==================================================

st.title("Crypto AutoHedge Lab")

st.caption(
    "Aggressive research strategy · Automatic paper execution "
    "· No real exchange orders"
)

st.divider()

# ==================================================
# 10. SCANNER SETTINGS
# ==================================================

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

# ==================================================
# 11. AUTOMATION CONTROL
# ==================================================

st.subheader("Automatic Paper Execution")

st.session_state.auto_enabled = st.toggle(
    "Enable automatic paper trading",
    value=st.session_state.auto_enabled,
    help=(
        "When enabled, qualifying signals can open or close "
        "virtual positions when the app processes market data."
    ),
)

if st.session_state.auto_enabled:
    st.success("Automatic paper execution is enabled.")
else:
    st.warning(
        "Automatic execution is paused. "
        "Existing virtual positions remain open."
    )

if not selected_symbols:
    st.warning("Select at least one asset.")
    st.stop()

# ==================================================
# 12. FETCH DATA
# ==================================================

with st.spinner("Loading public exchange market data..."):

    # Always fetch prices for open positions, even if they are
    # not currently selected in the Market Scanner.
    symbols_to_fetch = tuple(dict.fromkeys(
        list(selected_symbols)
        + list(st.session_state.positions.keys())
    ))

    market_data, market_errors = fetch_scanner_data(
        symbols_to_fetch,
        timeframe,
    )

# ==================================================
# 13. PROCESS AUTOMATIC TRADES
# ==================================================

execution_messages = []

if st.session_state.auto_enabled and timeframe == "15m":

    execution_messages = process_automatic_trades(
        market_data
    )

elif st.session_state.auto_enabled and timeframe != "15m":

    st.warning(
        "Automatic execution is configured for 15m only. "
        "Change the timeframe to 15m to enable automatic trades."
    )

if execution_messages:

    for message in execution_messages:
        st.success(message)

# ==================================================
# 14. MARKET OVERVIEW
# ==================================================

st.divider()
st.header("Market Overview")

if market_data:

    scanner_rows = []

    for symbol, data in market_data.items():

        candle = get_closed_candle(data["frame"])

        if candle is None:
            continue

        scanner_rows.append({
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

    if scanner_rows:

        st.dataframe(
            pd.DataFrame(scanner_rows),
            use_container_width=True,
            hide_index=True,
        )

else:

    st.warning(
        "No market data available. "
        "The exchanges may be temporarily unavailable."
    )

if market_errors:

    with st.expander("Market data errors"):

        for symbol, error in market_errors.items():
            st.write(f"**{symbol}:** {error}")

# ==================================================
# 15. ASSET ANALYSIS
# ==================================================

st.divider()
st.header("Asset Analysis")

available_symbols = list(market_data.keys())

if available_symbols:

    selected_asset = st.selectbox(
        "Select asset",
        available_symbols,
    )

    asset_data = market_data[selected_asset]
    frame = asset_data["frame"]
    candle = get_closed_candle(frame)

    if candle is not None:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current Price",
            f"${asset_data['price']:,.4f}",
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

        st.caption(
            "Indicators use the last completed candle. "
            "The displayed price is the latest available ticker price."
        )

# ==================================================
# 16. ACCOUNT OVERVIEW
# ==================================================

st.divider()
st.header("Account Overview")

equity = calculate_equity(market_data)

a1, a2, a3, a4 = st.columns(4)

a1.metric(
    "Starting Cash",
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
    "Paper Trades",
    len(st.session_state.trades),
)

# ==================================================
# 17. RISK MONITOR
# ==================================================

st.divider()
st.header("Risk Monitor")

open_positions = {
    symbol: float(quantity)
    for symbol, quantity in st.session_state.positions.items()
    if float(quantity) > 0
}

if open_positions:

    st.caption(
        "Live position values use the latest available public ticker price. "
        "P/L is estimated and excludes any future exit fee or slippage."
    )

    for symbol, quantity in open_positions.items():

        data = market_data.get(symbol)
        meta = st.session_state.position_meta.get(symbol, {})

        entry = meta.get("entry_price")
        entry = float(entry) if entry is not None else None

        current_price = (
            float(data["price"])
            if data and data.get("price") is not None
            else entry
        )

        stop_price = meta.get("stop_price")
        target_price = meta.get("target_price")

        unrealised_pnl = None
        unrealised_pct = None

        if entry is not None and entry > 0 and current_price is not None:
            unrealised_pnl = (current_price - entry) * quantity
            unrealised_pct = ((current_price - entry) / entry) * 100

        with st.container(border=True):

            st.subheader(f"🪙 {symbol}")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Entry Price",
                f"${entry:,.6f}" if entry is not None else "Unavailable",
            )

            c2.metric(
                "Current Price",
                f"${current_price:,.6f}" if current_price is not None else "Unavailable",
            )

            c3.metric(
                "Quantity Held",
                f"{quantity:.8f}",
            )

            c4, c5, c6 = st.columns(3)

            c4.metric(
                "Stop Loss",
                f"${float(stop_price):,.6f}" if stop_price is not None else "Not set",
            )

            c5.metric(
                "Take Profit",
                f"${float(target_price):,.6f}" if target_price is not None else "Not set",
            )

            if unrealised_pnl is not None:
                pnl_delta = f"{unrealised_pct:+.2f}%"
                c6.metric(
                    "Unrealized P/L",
                    f"${unrealised_pnl:+,.2f} USDT",
                    delta=pnl_delta,
                )
            else:
                c6.metric(
                    "Unrealized P/L",
                    "Unavailable",
                    help="This position has no saved entry price. Its P/L cannot be calculated accurately.",
                )

            if entry is None or stop_price is None or target_price is None:
                st.warning(
                    "This position is missing saved entry/stop/target details. "
                    "It may be an older position created before automatic risk metadata was added. "
                    "The app will not invent these values."
                )

else:

    st.info("No open virtual positions.")

# ==================================================
# 18. PAPER TRADE HISTORY

st.divider()
st.header("Paper Trade History")

if st.session_state.trades:

    st.dataframe(
        pd.DataFrame(st.session_state.trades),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.caption("No paper trades recorded yet.")

# ==================================================
# 19. STRATEGY SETTINGS
# ==================================================

st.divider()
st.header("Strategy Configuration")

settings = {
    "Strategy": "Aggressive",
    "Execution": "Automatic paper trading",
    "Risk per trade": "2% of estimated equity",
    "Stop loss": "1.5 × ATR14",
    "Take profit": "3 × ATR14",
    "Risk/reward target": "1:2",
    "Maximum allocation": "25% of estimated equity",
    "Simulated fee": "0.1% per side",
    "Entry timeframe": "15m",
}

st.dataframe(
    pd.DataFrame(
        list(settings.items()),
        columns=["Setting", "Value"],
    ),
    use_container_width=True,
    hide_index=True,
)

# ==================================================
# 20. FOOTER
# ==================================================

st.caption(
    "Educational paper-trading simulator only. "
    "Signals are rule-based and are not predictions. "
    "Public exchange data can be delayed or unavailable. "
    "Simulated fills do not represent guaranteed execution. "
    "Fees are approximated; slippage and funding are not included. "
    "Virtual balances are session-based and may reset."
)
