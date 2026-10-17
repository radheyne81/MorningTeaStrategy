#!/usr/bin/env python3
"""
Streamlit Cloud Mobile & Web App: NSE F&O Hourly CPR + 9:15 AM ATM±3 O=H/O=L Scanner
====================================================================================
Deploy this repository for FREE on Streamlit Community Cloud (https://share.streamlit.io)
to get a permanent mobile & desktop web link with live Yahoo Finance NSE data.
"""

import dataclasses
from datetime import datetime
import pandas as pd
import streamlit as st

from nse_fno_options_scanner import (
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
    page_title="NSE F&O Hourly CPR & 9:15 OHOL Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NSE F&O Tier-1 Options Scanner (Hourly CPR + 9:15 AM ATM±3 O=H/O=L)")
st.caption(
    "Intraday Hourly CPR (1H Pivot, BC, TC, Width %, R1/S1) • Strict F&O Ban & Mid-Cap Skipper • 9:15 AM Crabel Drive"
)

with st.sidebar:
    st.header("⚙️ Scanner & Risk Config")
    data_source = st.radio(
        "Data Feed Mode",
        ["Live Yahoo Finance (NSE)", "Verified Oct 16, 2026 Snapshot (Demo)"],
        index=0,
    )
    interval = st.selectbox("Intraday Candle Interval (for 1H CPR)", ["15m", "5m"], index=0)
    capital = st.number_input("Total Options Capital (₹)", min_value=25000, value=200000, step=25000)
    risk_pct = st.slider("Max Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    custom_ban_str = st.text_area(
        "F&O Ban Scrips to Skip (comma-separated)",
        value="ZEEL, BIOCON, PNB, SAIL, BANDHANBNK, HINDCOPPER, RBLBANK, MANAPPURAM, IEX, GNFC, CHAMBLFERT",
    )
    run_btn = st.button("🔄 Run Live Hourly CPR Scan", type="primary", use_container_width=True)

custom_bans = [s.strip() for s in custom_ban_str.split(",") if s.strip()]
eligible_tickers, ban_list_applied = filter_eligible_tier1_tickers(custom_bans)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Eligible Tier-1 Large-Caps", f"{len(eligible_tickers)} Scrips")
col2.metric("F&O Ban & Mid-Caps Skipped", f"{len(ban_list_applied)} Excluded")
col3.metric("Max Rupee Risk / Trade", f"₹{int(capital * risk_pct / 100):,}")
col4.metric("Scan Timestamp (IST)", datetime.now().strftime("%H:%M:%S"))

if data_source.startswith("Verified") or not run_btn:
    daily_map, intra_map = generate_demo_market_data(eligible_tickers)
else:
    with st.spinner(f"Fetching Live NSE 1D & {interval} OHLCV for {len(eligible_tickers)} Tier-1 Scrips..."):
        daily_map = fetch_yfinance_data(eligible_tickers, period="1y", interval="1d")
        intra_map = fetch_yfinance_data(eligible_tickers, period="5d", interval=interval)

open915_plans = []
intraday_plans = []
daily_plans = []

for sym in eligible_tickers:
    step, lot, sector, _ = TIER1_LIQUID_FNO_UNIVERSE[sym]
    if sym in daily_map and sym in intra_map:
        oplan = evaluate_915_opening_drive_and_atm_pm3(
            sym, daily_map[sym], intra_map[sym], step, lot, sector, capital, risk_pct
        )
        if oplan:
            open915_plans.append(dataclasses.asdict(oplan))
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

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ 9:15 AM + Hourly CPR + ATM±3 O=H/O=L",
    "🕒 Intraday Hourly CPR Scanner",
    "📅 Daily Swing Scanner",
    "📊 10-Day Historical Backtest",
])

with tab1:
    st.subheader("9:15 AM Opening Drive + 15m ORB + Hourly CPR + ATM±3 O=H/O=L Confluence")
    if open915_plans:
        st.dataframe(pd.DataFrame(open915_plans), use_container_width=True)
    else:
        st.info("No scrips currently meet all strict 9:15 AM + Hourly CPR + ATM±3 O=H/O=L filters.")

with tab2:
    st.subheader(f"Intraday ({interval}) Hourly CPR + ORB + VWAP Momentum Scanner")
    if intraday_plans:
        st.dataframe(pd.DataFrame(intraday_plans), use_container_width=True)
    else:
        st.info("No intraday Hourly CPR breakouts/breakdowns triggered right now.")

with tab3:
    st.subheader("Daily Swing / Positional Options Scanner (2–7 DTE)")
    if daily_plans:
        st.dataframe(pd.DataFrame(daily_plans), use_container_width=True)
    else:
        st.info("No daily swing setups matched.")

with tab4:
    st.subheader("10-Day Historical Backtest (Oct 5 – Oct 16, 2026) | Call & Put Setups")
    bt_records = [dataclasses.asdict(r) for r in run_10day_historical_backtest()]
    st.dataframe(pd.DataFrame(bt_records), use_container_width=True)
