from __future__ import annotations

from datetime import datetime, timezone

import ccxt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SafeDeal1 Crypto Lab", page_icon="📈", layout="wide")

STARTING_CASH = 10_000.00
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

st.title("Crypto AutoHedge Lab")
st.caption("Separate research workspace · Paper trading only · No real orders are sent")

if "cash" not in st.session_state:
    st.session_state.cash = STARTING_CASH
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "trades" not in st.session_state:
    st.session_state.trades = []

@st.cache_resource
def exchange_client():
    # Public market data only; no API keys or wallet secrets.
    return ccxt.binance({"enableRateLimit": True, "timeout": 10000})

@st.cache_data(ttl=30, show_spinner=False)
def fetch_prices():
    client = exchange_client()
    results = []
    for symbol in SYMBOLS:
        try:
            ticker = client.fetch_ticker(symbol)
            results.append({
                "Symbol": symbol,
                "Last price (USDT)": float(ticker.get("last") or 0),
                "24h change (%)": float(ticker.get("percentage") or 0),
                "24h quote volume": float(ticker.get("quoteVolume") or 0),
            })
        except Exception as exc:
            results.append({"Symbol": symbol, "Error": str(exc)[:160]})
    return results

try:
    market = fetch_prices()
except Exception as exc:
    market = [{"Symbol": "Market data", "Error": str(exc)[:200]}]

market_df = pd.DataFrame(market)
if "Last price (USDT)" in market_df.columns:
    left, mid, right = st.columns(3)
    left.metric("Virtual starting cash", f"${STARTING_CASH:,.2f}")
    mid.metric("Available virtual cash", f"${st.session_state.cash:,.2f}")
    right.metric("Recorded paper trades", len(st.session_state.trades))
    st.subheader("Public market prices")
    st.dataframe(market_df, use_container_width=True, hide_index=True)
else:
    st.warning("Market prices are temporarily unavailable. Check your internet connection or try again.")
    st.dataframe(market_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Paper trade")
st.info("These actions change only this browser session's virtual ledger. They do not send orders to an exchange.")
with st.form("paper_trade_form"):
    symbol = st.selectbox("Asset", SYMBOLS)
    side = st.radio("Action", ["Buy", "Sell"], horizontal=True)
    amount = st.number_input("Order value (USDT)", min_value=10.0, max_value=100_000.0, value=100.0, step=10.0)
    submitted = st.form_submit_button("Simulate paper trade", type="primary")

if submitted:
    price_row = next((r for r in market if r.get("Symbol") == symbol and r.get("Last price (USDT)", 0) > 0), None)
    if not price_row:
        st.error("A valid live market price is required to simulate this trade.")
    else:
        price = price_row["Last price (USDT)"]
        qty = amount / price
        positions = dict(st.session_state.positions)
        current_qty = float(positions.get(symbol, 0.0))
        if side == "Buy":
            if amount > st.session_state.cash:
                st.error("Not enough virtual cash for this paper trade.")
            else:
                positions[symbol] = current_qty + qty
                st.session_state.cash -= amount
                st.session_state.positions = positions
                st.session_state.trades.insert(0, {
                    "Time (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "Symbol": symbol, "Side": "BUY", "Quantity": qty,
                    "Price (USDT)": price, "Value (USDT)": amount, "Mode": "PAPER",
                })
                st.success(f"Simulated BUY: {qty:.8f} {symbol} at {price:,.4f} USDT.")
        else:
            if qty > current_qty:
                st.error(f"Not enough virtual {symbol} to sell. Current simulated quantity: {current_qty:.8f}.")
            else:
                positions[symbol] = current_qty - qty
                st.session_state.cash += amount
                st.session_state.positions = positions
                st.session_state.trades.insert(0, {
                    "Time (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "Symbol": symbol, "Side": "SELL", "Quantity": qty,
                    "Price (USDT)": price, "Value (USDT)": amount, "Mode": "PAPER",
                })
                st.success(f"Simulated SELL: {qty:.8f} {symbol} at {price:,.4f} USDT.")

st.divider()
st.subheader("Virtual positions")
position_rows = []
for asset, quantity in st.session_state.positions.items():
    price_row = next((r for r in market if r.get("Symbol") == asset and r.get("Last price (USDT)", 0) > 0), None)
    price = price_row["Last price (USDT)"] if price_row else None
    position_rows.append({
        "Symbol": asset,
        "Quantity": quantity,
        "Latest price (USDT)": price,
        "Market value (USDT)": quantity * price if price else None,
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

st.caption("Prices are public exchange data and may be delayed or unavailable. Paper trades omit fees, slippage, and order-book effects. This is not investment advice.")
