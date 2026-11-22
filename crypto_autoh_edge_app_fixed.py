from __future__ import annotations

from datetime import datetime, timezone

import ccxt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto AutoHedge Lab", page_icon="📈", layout="wide")

STARTING_CASH = 10_000.00
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
EXCHANGES = ["okx", "kucoin", "kraken"]  # Binance excluded: HTTP 451 in this deployment

st.title("Crypto AutoHedge Lab")
st.caption("Research workspace · Paper trading only · No real orders are sent")

if "cash" not in st.session_state:
    st.session_state.cash = STARTING_CASH
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "trades" not in st.session_state:
    st.session_state.trades = []

@st.cache_data(ttl=30, show_spinner=False)
def fetch_prices():
    errors = []
    for exchange_name in EXCHANGES:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            client = exchange_class({
                "enableRateLimit": True,
                "timeout": 20000,
            })
            markets = client.load_markets()
            rows = []

            for symbol in SYMBOLS:
                if symbol not in markets:
                    continue
                ticker = client.fetch_ticker(symbol)
                last = ticker.get("last")
                if last is None or float(last) <= 0:
                    continue
                rows.append({
                    "Symbol": symbol,
                    "Last price (USDT)": float(last),
                    "24h change (%)": float(ticker.get("percentage") or 0),
                    "24h quote volume": float(ticker.get("quoteVolume") or 0),
                    "Exchange": exchange_name.upper(),
                })

            if rows:
                return rows
            errors.append(f"{exchange_name}: no supported pairs")
        except Exception as exc:
            errors.append(f"{exchange_name}: {str(exc)[:180]}")

    return [{
        "Symbol": "Market data",
        "Error": "All public exchanges failed. " + " | ".join(errors),
    }]

market = fetch_prices()
market_df = pd.DataFrame(market)
has_prices = "Last price (USDT)" in market_df.columns

if has_prices:
    left, mid, right = st.columns(3)
    left.metric("Virtual starting cash", f"${STARTING_CASH:,.2f}")
    mid.metric("Available virtual cash", f"${st.session_state.cash:,.2f}")
    right.metric("Recorded paper trades", len(st.session_state.trades))
    st.subheader("Public market prices")
    st.dataframe(market_df, use_container_width=True, hide_index=True)
else:
    st.warning("Live market prices are unavailable from the supported public exchanges.")
    st.dataframe(market_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Paper trade")
st.info("These actions update only the virtual ledger in this app. They do not send exchange orders.")

with st.form("paper_trade_form"):
    symbol = st.selectbox("Asset", SYMBOLS)
    side = st.radio("Action", ["Buy", "Sell"], horizontal=True)
    amount = st.number_input(
        "Order value (USDT)",
        min_value=10.0,
        max_value=100_000.0,
        value=100.0,
        step=10.0,
    )
    submitted = st.form_submit_button("Simulate paper trade", type="primary")

if submitted:
    price_row = next(
        (row for row in market
         if row.get("Symbol") == symbol and row.get("Last price (USDT)", 0) > 0),
        None,
    )
    if price_row is None:
        st.error("A valid public market price is required for this paper simulation.")
    else:
        price = float(price_row["Last price (USDT)"])
        quantity = amount / price
        positions = dict(st.session_state.positions)
        current_quantity = float(positions.get(symbol, 0.0))

        if side == "Buy":
            if amount > st.session_state.cash:
                st.error("Not enough virtual cash for this paper trade.")
            else:
                positions[symbol] = current_quantity + quantity
                st.session_state.cash -= amount
                st.session_state.positions = positions
                st.session_state.trades.insert(0, {
                    "Time (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "Symbol": symbol,
                    "Side": "BUY",
                    "Quantity": quantity,
                    "Price (USDT)": price,
                    "Value (USDT)": amount,
                    "Mode": "PAPER",
                })
                st.success(f"Simulated BUY: {quantity:.8f} {symbol} at {price:,.4f} USDT.")
        else:
            if quantity > current_quantity:
                st.error(
                    f"Not enough virtual {symbol}. "
                    f"Current simulated quantity: {current_quantity:.8f}."
                )
            else:
                positions[symbol] = current_quantity - quantity
                st.session_state.cash += amount
                st.session_state.positions = positions
                st.session_state.trades.insert(0, {
                    "Time (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "Symbol": symbol,
                    "Side": "SELL",
                    "Quantity": quantity,
                    "Price (USDT)": price,
                    "Value (USDT)": amount,
                    "Mode": "PAPER",
                })
                st.success(f"Simulated SELL: {quantity:.8f} {symbol} at {price:,.4f} USDT.")

st.divider()
st.subheader("Virtual positions")
position_rows = []
for asset, quantity in st.session_state.positions.items():
    price_row = next(
        (row for row in market
         if row.get("Symbol") == asset and row.get("Last price (USDT)", 0) > 0),
        None,
    )
    price = float(price_row["Last price (USDT)"]) if price_row else None
    position_rows.append({
        "Symbol": asset,
        "Quantity": quantity,
        "Latest price (USDT)": price,
        "Market value (USDT)": quantity * price if price is not None else None,
    })

if position_rows:
    st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No simulated positions yet.")

st.subheader("Paper trade history")
if st.session_state.trades:
    st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True, hide_index=True)
else:
    st.caption("No paper trades recorded in this session.")

st.caption(
    "Market data comes from public exchange endpoints and may be unavailable. "
    "Paper simulations omit fees, slippage, and order-book effects. "
    "This is an educational research tool, not investment advice."
)
