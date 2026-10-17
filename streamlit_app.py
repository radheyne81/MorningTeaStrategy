#!/usr/bin/env python3
"""
Streamlit Cloud Automated Broker API Dashboard:
NSE F&O Hourly CPR + RSI(14) + Volatility Trigger + 9:15 AM ATM±3 O=H/O=L Scanner
"""

import dataclasses
from datetime import datetime
import time
import pandas as pd
import streamlit as st

from nse_fno_options_scanner import (
    BrokerAPIClient,
    TIER1_LIQUID_FNO_UNIVERSE,
    evaluate_915_opening_drive_and_atm_pm3,
    evaluate_daily_setup,
    evaluate_intraday_setup,
    fetch_yfinance_data,
    filter_eligible_tier1_tickers,
    generate_demo_market_data,
    run_10day_historical_backtest,
)

st.set_page_config(
    page_title="NSE F&O Automated Broker API Scanner (1H CPR + RSI 14 + Vol Trigger)",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ NSE F&O Automated Broker API Scanner (Hourly CPR + RSI 14 + Volatility Trigger)")
st.caption(
    "Angel One SmartAPI • DhanHQ v2 • Zerodha Kite v3 • Upstox API • Fast Auto-Refresh • RSI(14) ≥60/≤40 • BB/KC Squeeze + RVOL Volatility Trigger"
)

with st.sidebar:
    st.header("🤖 Broker API & Auto-Refresh")
    broker_choice = st.selectbox(
        "Broker API Provider",
        ["angelone", "dhan", "kite", "upstox", "yfinance"],
        index=0,
    )
    auto_trade_mode = st.selectbox(
        "Automated Execution Mode",
        ["paper", "live", "off"],
        index=0,
    )
    refresh_sec = st.selectbox(
        "Fast Auto-Refresh Interval (Seconds)",
        [0, 3, 5, 15, 30, 60],
        index=1,
        format_func=lambda x: "Manual (Paused)" if x == 0 else f"Every {x} Seconds",
    )
    data_source = st.radio(
        "Market Data Feed",
        ["Live Broker / Yahoo Finance", "Verified Oct 16, 2026 Snapshot (Demo)"],
        index=0,
    )
    interval = st.selectbox("Intraday Candle Interval (for 1H CPR)", ["15m", "5m"], index=0)
    capital = st.number_input("Total Options Capital (₹)", min_value=25000, value=200000, step=25000)
    risk_pct = st.slider("Max Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    custom_ban_str = st.text_area(
        "F&O Ban Scrips to Skip",
        value="ZEEL, BIOCON, PNB, SAIL, BANDHANBNK, HINDCOPPER, RBLBANK, MANAPPURAM, IEX, GNFC, CHAMBLFERT",
    )
    run_btn = st.button("⚡ Sync Broker API Now", type="primary", use_container_width=True)

custom_bans = [s.strip() for s in custom_ban_str.split(",") if s.strip()]
eligible_tickers, ban_list_applied = filter_eligible_tier1_tickers(custom_bans)
broker_client = BrokerAPIClient(broker_name=broker_choice, auto_trade_mode=auto_trade_mode)

if data_source.startswith("Verified"):
    daily_map, intra_map = generate_demo_market_data(eligible_tickers)
else:
    with st.spinner(f"Streaming {broker_choice.upper()} / NSE 1D & {interval} OHLCV..."):
        daily_map = fetch_yfinance_data(eligible_tickers, period="1y", interval="1d")
        intra_map = fetch_yfinance_data(eligible_tickers, period="5d", interval=interval)

open915_plans = []
intraday_plans = []
daily_plans = []

for sym in eligible_tickers:
    step, lot, sector, _ = TIER1_LIQUID_FNO_UNIVERSE[sym]
    if sym in daily_map and sym in intra_map:
        oplan = evaluate_915_opening_drive_and_atm_pm3(
            sym, daily_map[sym], intra_map[sym], step, lot, sector, capital, risk_pct, broker_client=broker_client
        )
        if oplan:
            open915_plans.append(dataclasses.asdict(oplan))
            broker_client.place_automated_option_order(oplan)
    if sym in intra_map:
        iplan = evaluate_intraday_setup(
            sym, intra_map[sym], step, lot, sector, capital, risk_pct
        )
        if iplan:
            intraday_plans.append(dataclasses.asdict(iplan))
    if sym in daily_map:
        dplan = evaluate_daily_setup(
            sym, daily_map[sym], step, lot, sector, capital, risk_pct
        )
        if dplan:
            daily_plans.append(dataclasses.asdict(dplan))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Broker API Gateway", f"{broker_choice.upper()} ({auto_trade_mode.upper()})")
col2.metric("Tier-1 Eligible / Skipped", f"{len(eligible_tickers)} / {len(ban_list_applied)} Ban")
col3.metric("Auto-Orders Fired", f"{len(broker_client.executed_orders)} Orders")
col4.metric("Last Fast Sync (IST)", datetime.now().strftime("%H:%M:%S"))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ 9:15 AM + 1H CPR + RSI(14) + Vol Blast",
    "🤖 Automated Broker Order Book",
    "🕒 Intraday 1H CPR + RSI(14) Scanner",
    "📅 Daily Swing Scanner",
    "📊 10-Day Historical Backtest",
])

with tab1:
    st.subheader("9:15 AM Opening Drive + 1H CPR + RSI(14) ≥60/≤40 + Volatility Blast + ATM±3 O=H/O=L")
    if open915_plans:
        st.dataframe(pd.DataFrame(open915_plans), use_container_width=True)
    else:
        st.info("No scrips currently meet all strict confluence filters.")

with tab2:
    st.subheader("Automated Broker API Order Book (Bracket SL at 1H CPR)")
    if broker_client.executed_orders:
        st.dataframe(pd.DataFrame([dataclasses.asdict(o) for o in broker_client.executed_orders]), use_container_width=True)
    else:
        st.info("No automated orders fired in this pass.")

with tab3:
    st.subheader(f"Intraday ({interval}) Hourly CPR + RSI(14) + Volatility Trigger Scanner")
    if intraday_plans:
        st.dataframe(pd.DataFrame(intraday_plans), use_container_width=True)
    else:
        st.info("No intraday Hourly CPR breakouts triggered right now.")

with tab4:
    st.subheader("Daily Swing / Positional Options Scanner (2–7 DTE)")
    if daily_plans:
        st.dataframe(pd.DataFrame(daily_plans), use_container_width=True)
    else:
        st.info("No daily swing setups matched.")

with tab5:
    st.subheader("10-Day Historical Backtest (Oct 5 – Oct 16, 2026)")
    bt_records = [dataclasses.asdict(r) for r in run_10day_historical_backtest()]
    st.dataframe(pd.DataFrame(bt_records), use_container_width=True)

if refresh_sec > 0:
    time.sleep(refresh_sec)
    st.rerun()
