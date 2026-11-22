
from __future__ import annotations

from datetime import datetime, timezone

import ccxt
import pandas as pd
import streamlit as st

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Crypto AutoHedge Lab",
    page_icon="📈",
    layout="wide",
)

STARTING_CASH = 10_000.00

DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]

EXCHANGES = ["okx", "kucoin", "kraken"]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------

st.title("Crypto AutoHedge Lab")

st.caption(
    "Research workspace · Paper trading only · "
    "No real exchange orders are sent"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "cash" not in st.session_state:
    st.session_state.cash = STARTING_CASH

if "positions" not in st.session_state:
    st.session_state.positions = {}

if "trades" not in st.session_state:
    st.session_state.trades = []

# --------------------------------------------------
# EXCHANGE CONNECTION
# --------------------------------------------------

@st.cache_resource
def get_exchange(name: str):
    exchange_class = getattr(ccxt, name)

    return exchange_class({
        "enableRateLimit": True,
        "timeout": 20000,
    })


# --------------------------------------------------
# MARKET DATA + TECHNICAL INDICATORS
# --------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_scanner_data(
    symbols: tuple[str, ...],
    timeframe: str,
    limit: int = 200,
):
    """
    Fetch public OHLCV and ticker data.
    Try OKX, KuCoin and Kraken in order.
    Calculate EMA20, EMA50, RSI14 and ATR14.
    """

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

                # Create DataFrame
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
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]:
                    frame[column] = pd.to_numeric(
                        frame[column],
                        errors="coerce",
                    )

                frame = frame.dropna(
                    subset=["Open", "High", "Low", "Close"]
                ).reset_index(drop=True)

                if frame.empty:
                    symbol_errors.append(
                        f"{exchange_name}: invalid candle data"
                    )
                    continue

                # EMA 20
                frame["EMA20"] = frame["Close"].ewm(
                    span=20,
                    adjust=False,
                ).mean()

                # EMA 50
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
                        (
                            frame["High"] - previous_close
                        ).abs(),
                        (
                            frame["Low"] - previous_close
                        ).abs(),
                    ],
                    axis=1,
                ).max(axis=1)

                frame["ATR14"] = true_range.ewm(
                    alpha=1 / 14,
                    adjust=False,
                    min_periods=14,
                ).mean()

                # Latest market price
                last_price = (
                    ticker.get("last")
                    or frame["Close"].iloc[-1]
                )

                if last_price is None or float(last_price) <= 0:
                    symbol_errors.append(
                        f"{exchange_name}: invalid ticker price"
                    )
                    continue

                # Save successful result
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
                    f"{exchange_name}: {str(exc)[:180]}"
                )
                continue

        if not found:
            errors[symbol] = " | ".join(symbol_errors)

    return output, errors


# --------------------------------------------------
# SIGNAL CLASSIFICATION
# --------------------------------------------------

def classify_signal(frame):

    if frame.empty:
        return "INSUFFICIENT DATA"

    latest = frame.iloc[-1]

    required = [
        "Close",
        "EMA20",
        "EMA50",
        "RSI14",
        "ATR14",
    ]

    if latest[required].isna().any():
        return "INSUFFICIENT DATA"

    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI14"])

    if rsi >= 70 or rsi <= 30:
        return "CAUTION"

    if close > ema20 > ema50 and 50 <= rsi < 70:
        return "BULLISH WATCH"

    if close < ema20 < ema50 and 30 < rsi <= 50:
        return "BEARISH WATCH"

    return "NEUTRAL"


# --------------------------------------------------
# MARKET SCANNER CONTROLS
# --------------------------------------------------

st.divider()

st.header("🔎 Market Scanner")

col1, col2 = st.columns(2)

with col1:
    selected_symbols = st.multiselect(
        "Select assets",
        options=DEFAULT_SYMBOLS,
        default=DEFAULT_SYMBOLS,
    )

with col2:
    timeframe = st.selectbox(
        "Primary timeframe",
        TIMEFRAMES,
        index=1,
    )

if not selected_symbols:
    st.warning("Please select at least one asset.")
    st.stop()

# --------------------------------------------------
# FETCH MARKET DATA
# --------------------------------------------------

with st.spinner("Fetching public market data..."):

    market_data, market_errors = fetch_scanner_data(
        tuple(selected_symbols),
        timeframe,
    )

# --------------------------------------------------
# MARKET OVERVIEW
# --------------------------------------------------

st.subheader("Market Overview")

if market_data:

    scanner_rows = []

    for symbol, data in market_data.items():

        frame = data["frame"]
        latest = frame.iloc[-1]

        scanner_rows.append({
            "Symbol": symbol,
            "Price (USDT)": data["price"],
            "24h Change (%)": data["change"],
            "EMA20": latest["EMA20"],
            "EMA50": latest["EMA50"],
            "RSI14": latest["RSI14"],
            "ATR14": latest["ATR14"],
            "Signal": classify_signal(frame),
            "Exchange": data["exchange"],
        })

    scanner_df = pd.DataFrame(scanner_rows)

    st.dataframe(
        scanner_df,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning(
        "Market data is currently unavailable from supported exchanges."
    )

if market_errors:

    with st.expander("Market data errors"):

        for symbol, error in market_errors.items():
            st.write(f"**{symbol}:** {error}")

# --------------------------------------------------
# INDIVIDUAL ASSET ANALYSIS
# --------------------------------------------------

st.subheader("Asset Analysis")

available_symbols = list(market_data.keys())

if available_symbols:

    selected_asset = st.selectbox(
        "Choose asset to analyse",
        available_symbols,
    )

    asset_data = market_data[selected_asset]
    frame = asset_data["frame"]
    latest = frame.iloc[-1]

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Latest Price",
        f"${asset_data['price']:,.4f}",
    )

    metric2.metric(
        "EMA20",
        f"${latest['EMA20']:,.4f}",
    )

    metric3.metric(
        "EMA50",
        f"${latest['EMA50']:,.4f}",
    )

    metric4.metric(
        "RSI14",
        (
            f"{latest['RSI14']:.2f}"
            if pd.notna(latest["RSI14"])
            else "N/A"
        ),
    )

    st.write(
        f"**Scanner signal:** {classify_signal(frame)}"
    )

    st.write("**Price and EMA chart**")

    chart_frame = frame.set_index("Timestamp")[
        ["Close", "EMA20", "EMA50"]
    ]

    st.line_chart(chart_frame)

    st.write("**Recent candles and indicators**")

    st.dataframe(
        frame.tail(20).sort_values(
            "Timestamp",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# ACCOUNT OVERVIEW
# --------------------------------------------------

st.divider()

st.header("💰 Account Overview")

account1, account2, account3 = st.columns(3)

account1.metric(
    "Virtual Starting Cash",
    f"${STARTING_CASH:,.2f}",
)

account2.metric(
    "Available Virtual Cash",
    f"${st.session_state.cash:,.2f}",
)

account3.metric(
    "Recorded Paper Trades",
    len(st.session_state.trades),
)

# --------------------------------------------------
# PAPER TRADING
# --------------------------------------------------

st.subheader("Paper Trade")

st.info(
    "These actions update only your virtual ledger. "
    "No real exchange orders are sent."
)

with st.form("paper_trade_form"):

    trade_symbol = st.selectbox(
        "Asset",
        DEFAULT_SYMBOLS,
    )

    side = st.radio(
        "Action",
        ["Buy", "Sell"],
        horizontal=True,
    )

    amount = st.number_input(
        "Order value (USDT)",
        min_value=10.0,
        max_value=100_000.0,
        value=100.0,
        step=10.0,
    )

    submitted = st.form_submit_button(
        "Simulate paper trade",
        type="primary",
    )

if submitted:

    price_data = market_data.get(trade_symbol)

    if price_data is None:

        st.error(
            "A valid public market price is required "
            "for this paper simulation."
        )

    else:

        price = float(price_data["price"])
        quantity = amount / price

        positions = dict(st.session_state.positions)

        current_quantity = float(
            positions.get(trade_symbol, 0.0)
        )

        if side == "Buy":

            if amount > st.session_state.cash:

                st.error(
                    "Not enough virtual cash for this paper trade."
                )

            else:

                positions[trade_symbol] = (
                    current_quantity + quantity
                )

                st.session_state.cash -= amount
                st.session_state.positions = positions

                st.session_state.trades.insert(0, {
                    "Time (UTC)": datetime.now(
                        timezone.utc
                    ).isoformat(timespec="seconds"),
                    "Symbol": trade_symbol,
                    "Side": "BUY",
                    "Quantity": quantity,
                    "Price (USDT)": price,
                    "Value (USDT)": amount,
                    "Mode": "PAPER",
                })

                st.success(
                    f"Simulated BUY: {quantity:.8f} "
                    f"{trade_symbol} at {price:,.4f} USDT."
                )

        else:

            if quantity > current_quantity:

                st.error(
                    f"Not enough virtual {trade_symbol}. "
                    f"Current simulated quantity: "
                    f"{current_quantity:.8f}."
                )

            else:

                positions[trade_symbol] = (
                    current_quantity - quantity
                )

                st.session_state.cash += amount
                st.session_state.positions = positions

                st.session_state.trades.insert(0, {
                    "Time (UTC)": datetime.now(
                        timezone.utc
                    ).isoformat(timespec="seconds"),
                    "Symbol": trade_symbol,
                    "Side": "SELL",
                    "Quantity": quantity,
                    "Price (USDT)": price,
                    "Value (USDT)": amount,
                    "Mode": "PAPER",
                })

                st.success(
                    f"Simulated SELL: {quantity:.8f} "
                    f"{trade_symbol} at {price:,.4f} USDT."
                )

# --------------------------------------------------
# RISK MONITOR
# --------------------------------------------------

st.divider()

st.header("🛡️ Risk Monitor")

position_rows = []

for asset, quantity in st.session_state.positions.items():

    price_data = market_data.get(asset)

    price = (
        float(price_data["price"])
        if price_data is not None
        else None
    )

    position_rows.append({
        "Symbol": asset,
        "Quantity": quantity,
        "Latest Price (USDT)": price,
        "Market Value (USDT)": (
            quantity * price
            if price is not None
            else None
        ),
    })

if position_rows:

    position_df = pd.DataFrame(position_rows)

    st.dataframe(
        position_df,
        use_container_width=True,
        hide_index=True,
    )

    known_market_values = [
        row["Market Value (USDT)"]
        for row in position_rows
        if row["Market Value (USDT)"] is not None
    ]

    total_market_value = sum(known_market_values)

    total_equity = (
        st.session_state.cash + total_market_value
    )

    st.metric(
        "Estimated Virtual Account Equity",
        f"${total_equity:,.2f}",
    )

else:

    st.caption("No simulated positions yet.")

# --------------------------------------------------
# PAPER TRADE HISTORY
# --------------------------------------------------

st.divider()

st.subheader("Paper Trade History")

if st.session_state.trades:

    st.dataframe(
        pd.DataFrame(st.session_state.trades),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.caption("No paper trades recorded in this session.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Market data comes from public exchange endpoints and "
    "may be unavailable. Scanner signals use simple indicator "
    "rules and are not predictions. Paper simulations omit fees, "
    "slippage and order-book effects. Educational research only."
)
