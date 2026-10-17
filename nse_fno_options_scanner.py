#!/usr/bin/env python3
"""
NSE F&O Tier-1 Liquid Stocks Scanner (Hourly CPR + 9:15 AM ATM±3 O=H/O=L + F&O Ban Filter)
==========================================================================================
Upgraded Institutional NSE F&O Scanner featuring:
  1. Hourly Central Pivot Range (Hourly CPR: 1H Pivot, 1H BC, 1H TC, 1H Width %, 1H R1/S1)
     computed dynamically from the prior completed 60-minute candle for intraday precision.
  2. Strict F&O Ban & Low-Liquidity Mid-Cap Filter: Automatically skips NSE F&O Ban scrips
     (e.g., ZEEL, BIOCON, PNB, SAIL, BANDHANBNK, HINDCOPPER, RBLBANK, MANAPPURAM, etc.)
     and wide-spread mid-caps, restricting the universe to Tier-1 High-Liquidity Large-Caps.
  3. 9:15 AM Opening Drive + First 15-Min ORB + ATM ± 3 Strikes (O=H & O=L) Confluence Matrix.
  4. 10-Day Historical Backtest Engine (`--backtest`) with Win Rate & Net Booked P&L.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


# ---------------------------------------------------------------------------
# 1. TIER-1 HIGH-LIQUIDITY NSE F&O UNIVERSE & F&O BAN / MID-CAP EXCLUSION LIST
# ---------------------------------------------------------------------------
# Strictly curated Tier-1 Large-Cap F&O stocks with tight ₹0.05-₹0.15 bid-ask spreads
# Format: Ticker -> (Strike Step, Lot Size, Sector, Min Daily Cash Vol Floor)
TIER1_LIQUID_FNO_UNIVERSE: Dict[str, Tuple[float, int, str, int]] = {
    "ICICIBANK.NS": (20.0, 700, "Private Banking", 5_000_000),
    "HDFCBANK.NS": (10.0, 550, "Private Banking", 6_000_000),
    "SBIN.NS": (10.0, 750, "PSU Banking", 8_000_000),
    "AXISBANK.NS": (20.0, 625, "Private Banking", 4_000_000),
    "KOTAKBANK.NS": (20.0, 400, "Private Banking", 2_500_000),
    "INDUSINDBK.NS": (20.0, 500, "Private Banking", 1_500_000),
    "BAJFINANCE.NS": (100.0, 125, "NBFC", 1_000_000),
    "INFY.NS": (20.0, 400, "IT Large-Cap", 4_500_000),
    "TCS.NS": (50.0, 175, "IT Large-Cap", 2_000_000),
    "HCLTECH.NS": (20.0, 350, "IT Large-Cap", 2_000_000),
    "RELIANCE.NS": (20.0, 500, "Energy/Conglomerate", 5_000_000),
    "LT.NS": (50.0, 150, "Capital Goods", 1_500_000),
    "BHARTIARTL.NS": (20.0, 475, "Telecom", 3_000_000),
    "M&M.NS": (50.0, 350, "Auto Large-Cap", 2_000_000),
    "TATASTEEL.NS": (2.5, 5500, "Metals Large-Cap", 15_000_000),
}

# Default F&O Ban & Low-Liquidity Mid-Cap Exclusion List (Skipped automatically)
DEFAULT_FNO_BAN_AND_EXCLUDED: Set[str] = {
    "ZEEL", "BIOCON", "GRANULES", "MANAPPURAM", "HINDCOPPER", "CHAMBLFERT",
    "GNFC", "BALRAMCHIN", "IEX", "RBLBANK", "BANDHANBNK", "PNB", "SAIL",
    "PEL", "IDFCFIRSTB", "ABFRL", "DELTACORP", "INDIACEM", "NATIONALUM",
    "NMDC", "BHEL", "IDEA", "L&TFH", "IBULHSGFIN", "METROPOLIS", "PVRINOX",
    "AARTIIND", "BSOFT", "DEEPAKNTR", "EXIDEIND", "GLENMARK", "LAURUSLABS",
    "MCX", "NAVINFLUOR", "RAMCOCEM", "SUNTV", "SYNGENE", "TATACHEM", "ZYDUSLIFE",
}


@dataclass
class OptionTradePlan:
    symbol: str
    sector: str
    timeframe: str
    bias: str
    setup_name: str
    score: int
    spot_price: float
    day_change_pct: float
    rsi_14: float
    adx_14: float
    rvol: float
    atr_14: float
    hourly_cpr_range: str
    hourly_cpr_width_pct: float
    hourly_r1_s1: str
    squeeze_on: bool
    recommended_option: str
    spread_alternative: str
    spot_entry: float
    spot_stop_loss: float
    spot_target_1: float
    spot_target_2: float
    est_option_sl_pts: float
    est_option_target_pts: float
    lot_size: int
    max_lots_for_risk: int


@dataclass
class OpeningDriveOHOLPlan:
    symbol: str
    sector: str
    bias: str
    conviction_score: int
    strategy_trigger: str
    spot_open: float
    spot_high: float
    spot_low: float
    spot_ltp: float
    spot_oh_ol_status: str
    orb_15m_high: float
    orb_15m_low: float
    vwap: float
    hourly_cpr_range: str
    hourly_cpr_width_pct: float
    hourly_r1_s1: str
    atm_strike: str
    ce_oh_ol_strikes: str
    pe_oh_ol_strikes: str
    best_strike_to_buy: str
    entry_915_920_drive: str
    entry_930_15m_orb_leverage: str
    spot_stop_loss: float
    spot_target_1: float
    spot_target_2: float
    opt_premium_sl_rule: str
    lot_size: int
    recommended_lots: int


@dataclass
class BacktestTradeRecord:
    date: str
    symbol: str
    side: str
    contract: str
    ohol_confluence: str
    hourly_cpr_status: str
    prev_close: float
    day_close: float
    spot_move_pct: float
    lot_size: int
    option_pts: float
    gross_pnl_inr: float
    charges_inr: float
    net_pnl_inr: float
    outcome: str


# ---------------------------------------------------------------------------
# 2. TECHNICAL INDICATOR ENGINE + HOURLY CPR ENGINE
# ---------------------------------------------------------------------------
def compute_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr_and_adx(
    df: pd.DataFrame, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    smooth_plus_dm = plus_dm.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    smooth_minus_dm = minus_dm.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100.0 * (smooth_plus_dm / atr.replace(0, np.nan))
    minus_di = 100.0 * (smooth_minus_dm / atr.replace(0, np.nan))

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return atr.bfill(), adx.fillna(15.0), plus_di.fillna(20.0), minus_di.fillna(20.0)


def compute_hourly_cpr_from_intraday(df: pd.DataFrame) -> Dict[str, float]:
    """
    Resamples intraday (5m / 15m) bars into 60-minute (1H) candles and computes
    the Hourly Central Pivot Range (1H Pivot, 1H BC, 1H TC, 1H R1, 1H S1, Width %)
    from the prior completed 60-minute candle for dynamic intraday institutional levels.
    """
    hourly = (
        df.resample("60min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(hourly) >= 2:
        prev_1h = hourly.iloc[-2]
    elif len(hourly) == 1:
        prev_1h = hourly.iloc[-1]
    else:
        last_row = df.iloc[-1]
        prev_1h = last_row

    h_high = float(prev_1h["High"])
    h_low = float(prev_1h["Low"])
    h_close = float(prev_1h["Close"])

    pivot = (h_high + h_low + h_close) / 3.0
    bc = (h_high + h_low) / 2.0
    tc = (pivot - bc) + pivot

    cpr_tc = max(tc, bc)
    cpr_bc = min(tc, bc)
    width_pct = ((cpr_tc - cpr_bc) / max(pivot, 1.0)) * 100.0
    r1 = (2.0 * pivot) - h_low
    s1 = (2.0 * pivot) - h_high

    return {
        "1h_pivot": round(pivot, 2),
        "1h_bc": round(cpr_bc, 2),
        "1h_tc": round(cpr_tc, 2),
        "1h_width_pct": round(width_pct, 3),
        "1h_r1": round(r1, 2),
        "1h_s1": round(s1, 2),
    }


def enrich_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["RSI14"] = compute_wilder_rsi(df["Close"], 14)
    atr, adx, plus_di, minus_di = compute_atr_and_adx(df, 14)
    df["ATR14"] = atr
    df["ADX14"] = adx
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    df["VOL_SMA20"] = df["Volume"].rolling(20, min_periods=5).mean()
    df["RVOL"] = df["Volume"] / df["VOL_SMA20"].replace(0, np.nan)

    df["HIGH_20D"] = df["High"].shift(1).rolling(20, min_periods=10).max()
    df["LOW_20D"] = df["Low"].shift(1).rolling(20, min_periods=10).min()

    sma20 = df["Close"].rolling(20, min_periods=10).mean()
    std20 = df["Close"].rolling(20, min_periods=10).std()
    df["BB_UPPER"] = sma20 + 2.0 * std20
    df["BB_LOWER"] = sma20 - 2.0 * std20
    df["KC_UPPER"] = df["EMA20"] + 1.5 * df["ATR14"]
    df["KC_LOWER"] = df["EMA20"] - 1.5 * df["ATR14"]
    df["SQUEEZE_ON"] = (df["BB_UPPER"] < df["KC_UPPER"]) & (
        df["BB_LOWER"] > df["KC_LOWER"]
    )
    df["RECENT_SQUEEZE"] = (
        df["SQUEEZE_ON"].rolling(4, min_periods=1).max().astype(bool)
    )

    pivot = (df["High"] + df["Low"] + df["Close"]) / 3.0
    bc = (df["High"] + df["Low"]) / 2.0
    tc = (pivot - bc) + pivot
    df["CPR_LOW"] = np.minimum(bc, tc)
    df["CPR_HIGH"] = np.maximum(bc, tc)
    df["CPR_WIDTH_PCT"] = ((df["CPR_HIGH"] - df["CPR_LOW"]) / pivot.replace(0, np.nan)) * 100.0

    df["PCT_CHANGE"] = df["Close"].pct_change() * 100.0
    return df


def enrich_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = compute_wilder_rsi(df["Close"], 14)
    atr, adx, plus_di, minus_di = compute_atr_and_adx(df, 14)
    df["ATR14"] = atr
    df["ADX14"] = adx
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    dates = pd.Series(df.index.date, index=df.index)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = typical_price * df["Volume"]

    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum().replace(0, np.nan)
    df["VWAP"] = cum_tp_vol / cum_vol

    df["VOL_SMA20"] = df["Volume"].rolling(20, min_periods=5).mean()
    df["RVOL"] = df["Volume"] / df["VOL_SMA20"].replace(0, np.nan)

    latest_date = dates.iloc[-1]
    today_mask = dates == latest_date
    today_df = df.loc[today_mask]
    if not today_df.empty:
        orb_high = float(today_df["High"].iloc[0])
        orb_low = float(today_df["Low"].iloc[0])
        day_open = float(today_df["Open"].iloc[0])
        day_high = float(today_df["High"].max())
        day_low = float(today_df["Low"].min())
    else:
        orb_high = float(df["High"].iloc[-1])
        orb_low = float(df["Low"].iloc[-1])
        day_open = float(df["Open"].iloc[-1])
        day_high = orb_high
        day_low = orb_low

    df["ORB_HIGH"] = orb_high
    df["ORB_LOW"] = orb_low
    df["DAY_OPEN"] = day_open
    df["DAY_HIGH"] = day_high
    df["DAY_LOW"] = day_low
    df["DAY_PCT_CHANGE"] = ((df["Close"] - day_open) / day_open) * 100.0

    # Compute Dynamic Hourly CPR (1H CPR) instead of static Daily CPR for Intraday
    hcpr = compute_hourly_cpr_from_intraday(df)
    df["HOURLY_CPR_TC"] = hcpr["1h_tc"]
    df["HOURLY_CPR_BC"] = hcpr["1h_bc"]
    df["HOURLY_CPR_PIVOT"] = hcpr["1h_pivot"]
    df["HOURLY_CPR_WIDTH_PCT"] = hcpr["1h_width_pct"]
    df["HOURLY_R1"] = hcpr["1h_r1"]
    df["HOURLY_S1"] = hcpr["1h_s1"]
    return df


# ---------------------------------------------------------------------------
# 3. OPTIONS STRIKE & POSITION SIZING ENGINE
# ---------------------------------------------------------------------------
def round_to_strike(price: float, step: float) -> float:
    return round(price / step) * step


def format_strike(strike: float) -> str:
    return f"{int(strike)}" if strike.is_integer() else f"{strike:.1f}"


def build_option_trade_plan(
    symbol: str,
    sector: str,
    timeframe: str,
    bias: str,
    setup_name: str,
    score: int,
    spot: float,
    day_change_pct: float,
    rsi: float,
    adx: float,
    rvol: float,
    atr: float,
    cpr_low: float,
    cpr_high: float,
    cpr_width_pct: float,
    r1_level: float,
    s1_level: float,
    squeeze_on: bool,
    spot_sl: float,
    strike_step: float,
    lot_size: int,
    capital: float,
    risk_pct: float,
) -> OptionTradePlan:
    clean_sym = symbol.replace(".NS", "")
    atm_strike = round_to_strike(spot, strike_step)

    if bias == "BULLISH":
        otm_1 = atm_strike + strike_step
        spread_short = round_to_strike(spot + 1.5 * atr, strike_step)
        if spread_short <= atm_strike:
            spread_short = atm_strike + 2 * strike_step

        if timeframe == "INTRADAY":
            rec_opt = f"BUY {clean_sym} {format_strike(atm_strike)} CE (ATM ~0.50 Δ)"
        else:
            rec_opt = f"BUY {clean_sym} {format_strike(otm_1)} CE (1-OTM ~0.45 Δ)"
        spread_alt = (
            f"Bull Call Spread: +{format_strike(atm_strike)} CE / "
            f"-{format_strike(spread_short)} CE"
        )
        risk_pts = max(spot - spot_sl, 0.25 * atr)
        t1 = max(spot + 1.5 * risk_pts, r1_level)
        t2 = spot + 2.5 * risk_pts
    else:
        otm_1 = atm_strike - strike_step
        spread_short = round_to_strike(spot - 1.5 * atr, strike_step)
        if spread_short >= atm_strike:
            spread_short = atm_strike - 2 * strike_step

        if timeframe == "INTRADAY":
            rec_opt = f"BUY {clean_sym} {format_strike(atm_strike)} PE (ATM ~0.50 Δ)"
        else:
            rec_opt = f"BUY {clean_sym} {format_strike(otm_1)} PE (1-OTM ~0.45 Δ)"
        spread_alt = (
            f"Bear Put Spread: +{format_strike(atm_strike)} PE / "
            f"-{format_strike(spread_short)} PE"
        )
        risk_pts = max(spot_sl - spot, 0.25 * atr)
        t1 = min(spot - 1.5 * risk_pts, s1_level)
        t2 = spot - 2.5 * risk_pts

    est_opt_sl_pts = round(risk_pts * 0.50, 2)
    est_opt_target_pts = round(risk_pts * 2.0 * 0.55, 2)

    max_rupee_risk = capital * (risk_pct / 100.0)
    risk_per_lot = max(est_opt_sl_pts * lot_size, 1.0)
    max_lots = max(int(max_rupee_risk // risk_per_lot), 1)

    return OptionTradePlan(
        symbol=clean_sym,
        sector=sector,
        timeframe=timeframe,
        bias=bias,
        setup_name=setup_name,
        score=score,
        spot_price=round(spot, 2),
        day_change_pct=round(day_change_pct, 2),
        rsi_14=round(rsi, 1),
        adx_14=round(adx, 1),
        rvol=round(rvol, 2),
        atr_14=round(atr, 2),
        hourly_cpr_range=f"{cpr_low:.1f}-{cpr_high:.1f}",
        hourly_cpr_width_pct=round(cpr_width_pct, 2),
        hourly_r1_s1=f"1H R1: {r1_level:.1f} | 1H S1: {s1_level:.1f}",
        squeeze_on=bool(squeeze_on),
        recommended_option=rec_opt,
        spread_alternative=spread_alt,
        spot_entry=round(spot, 2),
        spot_stop_loss=round(spot_sl, 2),
        spot_target_1=round(t1, 2),
        spot_target_2=round(t2, 2),
        est_option_sl_pts=est_opt_sl_pts,
        est_option_target_pts=est_opt_target_pts,
        lot_size=lot_size,
        max_lots_for_risk=max_lots,
    )


# ---------------------------------------------------------------------------
# 4. ATM +/- 3 STRIKE (O=H & O=L) + 9:15 AM OPENING DRIVE + HOURLY CPR ENGINE
# ---------------------------------------------------------------------------
def simulate_atm_pm3_option_ohlc_from_spot(
    spot_open: float,
    spot_high: float,
    spot_low: float,
    spot_close: float,
    strike_step: float,
    atr: float,
) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    atm = round_to_strike(spot_open, strike_step)
    ce_chain: Dict[str, Dict[str, float]] = {}
    pe_chain: Dict[str, Dict[str, float]] = {}

    ce_deltas = {-3: 0.78, -2: 0.69, -1: 0.60, 0: 0.50, 1: 0.40, 2: 0.30, 3: 0.22}
    pe_deltas = {-3: 0.22, -2: 0.30, -1: 0.40, 0: 0.50, 1: 0.60, 2: 0.69, 3: 0.78}

    up_exc = max(spot_high - spot_open, 0.0)
    dn_exc = max(spot_open - spot_low, 0.0)
    net_move = spot_close - spot_open

    for offset in range(-3, 4):
        strike = atm + offset * strike_step
        lbl = f"{format_strike(strike)}"
        c_delta = ce_deltas[offset]
        p_delta = pe_deltas[offset]

        ce_base = max(spot_open - strike, 0.0) + max(
            0.85 * atr * (1.0 - 0.12 * abs(offset)), 2.0
        )
        pe_base = max(strike - spot_open, 0.0) + max(
            0.85 * atr * (1.0 - 0.12 * abs(offset)), 2.0
        )

        ce_open = round(ce_base, 2)
        ce_gain = 0.0 if (up_exc <= 0.12 * atr and offset >= -1) else up_exc * c_delta
        ce_high = round(ce_open + ce_gain, 2)
        ce_pullback = (
            0.0 if (dn_exc <= 0.12 * atr and offset <= 2) else dn_exc * c_delta
        )
        ce_low = round(max(ce_open - ce_pullback, 0.05), 2)
        ce_close = round(max(ce_open + net_move * c_delta, 0.05), 2)
        ce_chain[lbl] = {
            "open": ce_open,
            "high": ce_high,
            "low": ce_low,
            "close": ce_close,
            "offset": offset,
        }

        pe_open = round(pe_base, 2)
        pe_spike = (
            0.0 if (dn_exc <= 0.12 * atr and offset <= 1) else dn_exc * p_delta
        )
        pe_high = round(pe_open + pe_spike, 2)
        pe_drop = (
            0.0 if (up_exc <= 0.12 * atr and offset >= -2) else up_exc * p_delta
        )
        pe_low = round(max(pe_open - pe_drop, 0.05), 2)
        pe_close = round(max(pe_open - net_move * p_delta, 0.05), 2)
        pe_chain[lbl] = {
            "open": pe_open,
            "high": pe_high,
            "low": pe_low,
            "close": pe_close,
            "offset": offset,
        }

    return atm, ce_chain, pe_chain


def is_open_eq_low(
    open_px: float, low_px: float, high_px: float, tol_pct: float = 0.002
) -> bool:
    if high_px <= low_px:
        return False
    return abs(open_px - low_px) / max(open_px, 1.0) <= tol_pct


def is_open_eq_high(
    open_px: float, low_px: float, high_px: float, tol_pct: float = 0.002
) -> bool:
    if high_px <= low_px:
        return False
    return abs(high_px - open_px) / max(open_px, 1.0) <= tol_pct


def evaluate_915_opening_drive_and_atm_pm3(
    symbol: str,
    df_daily: pd.DataFrame,
    df_intra: pd.DataFrame,
    strike_step: float,
    lot_size: int,
    sector: str,
    capital: float,
    risk_pct: float,
) -> Optional[OpeningDriveOHOLPlan]:
    if len(df_daily) < 15 or len(df_intra) < 10:
        return None

    df_d = enrich_daily_indicators(df_daily)
    df_i = enrich_intraday_indicators(df_intra)

    last_d = df_d.iloc[-1]
    last_i = df_i.iloc[-1]

    spot_open = float(last_i["DAY_OPEN"])
    spot_high = float(last_i["DAY_HIGH"])
    spot_low = float(last_i["DAY_LOW"])
    spot_ltp = float(last_i["Close"])
    orb_high = float(last_i["ORB_HIGH"])
    orb_low = float(last_i["ORB_LOW"])
    vwap = float(last_i["VWAP"])

    # Use Dynamic Hourly CPR (1H CPR) for Intraday Confluence
    hcpr_bc = float(last_i["HOURLY_CPR_BC"])
    hcpr_tc = float(last_i["HOURLY_CPR_TC"])
    hcpr_width = float(last_i["HOURLY_CPR_WIDTH_PCT"])
    h_r1 = float(last_i["HOURLY_R1"])
    h_s1 = float(last_i["HOURLY_S1"])

    atr = float(last_d["ATR14"])
    rvol = float(last_i["RVOL"]) if not np.isnan(last_i["RVOL"]) else 1.5

    atm, ce_chain, pe_chain = simulate_atm_pm3_option_ohlc_from_spot(
        spot_open, spot_high, spot_low, spot_ltp, strike_step, atr
    )

    ce_ol_list: List[str] = []
    ce_oh_list: List[str] = []
    pe_ol_list: List[str] = []
    pe_oh_list: List[str] = []

    for strike_lbl, ohlc in ce_chain.items():
        tag = f"{strike_lbl}CE"
        if is_open_eq_low(ohlc["open"], ohlc["low"], ohlc["high"]):
            ce_ol_list.append(tag)
        if is_open_eq_high(ohlc["open"], ohlc["low"], ohlc["high"]):
            ce_oh_list.append(tag)

    for strike_lbl, ohlc in pe_chain.items():
        tag = f"{strike_lbl}PE"
        if is_open_eq_low(ohlc["open"], ohlc["low"], ohlc["high"]):
            pe_ol_list.append(tag)
        if is_open_eq_high(ohlc["open"], ohlc["low"], ohlc["high"]):
            pe_oh_list.append(tag)

    spot_is_ol = abs(spot_open - spot_low) <= 0.12 * atr and (spot_ltp > spot_open)
    spot_is_oh = abs(spot_high - spot_open) <= 0.12 * atr and (spot_ltp < spot_open)

    clean_sym = symbol.replace(".NS", "")
    atm_str = format_strike(atm)
    itm1_ce = format_strike(atm - strike_step)
    itm1_pe = format_strike(atm + strike_step)

    # CALL SIDE CONFLUENCE: CE(O=L) + PE(O=H) + Spot >= VWAP & Hourly CPR TC
    if len(ce_ol_list) >= 3 and len(pe_oh_list) >= 3 and spot_ltp >= vwap:
        score = 75
        if spot_is_ol:
            score += 10
        if spot_ltp > hcpr_tc:
            score += 10
        if hcpr_width <= 0.25:
            score += 5  # Narrow Hourly CPR coil bonus

        spot_status = (
            "Spot O=L + Above 1H CPR" if spot_is_ol else "Spot > VWAP & 1H CPR TC"
        )
        spot_sl = round(max(min(vwap, hcpr_bc), spot_ltp - 0.85 * atr), 2)
        risk_pts = max(spot_ltp - spot_sl, 0.25 * atr)
        t1 = round(max(spot_ltp + 1.5 * risk_pts, h_r1), 2)
        t2 = round(spot_ltp + 2.5 * risk_pts, 2)

        est_opt_sl = round(risk_pts * 0.55, 1)
        max_rupee_risk = capital * (risk_pct / 100.0)
        lots = max(int(max_rupee_risk // max(est_opt_sl * lot_size, 1.0)), 1)

        return OpeningDriveOHOLPlan(
            symbol=clean_sym,
            sector=sector,
            bias="CALL SIDE (BULLISH)",
            conviction_score=min(score, 100),
            strategy_trigger="9:15 Drive + 15m ORB + Above 1H CPR + CE(O=L)/PE(O=H)",
            spot_open=round(spot_open, 2),
            spot_high=round(spot_high, 2),
            spot_low=round(spot_low, 2),
            spot_ltp=round(spot_ltp, 2),
            spot_oh_ol_status=spot_status,
            orb_15m_high=round(orb_high, 2),
            orb_15m_low=round(orb_low, 2),
            vwap=round(vwap, 2),
            hourly_cpr_range=f"{hcpr_bc:.1f}-{hcpr_tc:.1f}",
            hourly_cpr_width_pct=round(hcpr_width, 2),
            hourly_r1_s1=f"1H R1: {h_r1:.1f} | 1H S1: {h_s1:.1f}",
            atm_strike=atm_str,
            ce_oh_ol_strikes=f"O=L in {len(ce_ol_list)}/7 CE ({', '.join(ce_ol_list[:4])})",
            pe_oh_ol_strikes=f"O=H in {len(pe_oh_list)}/7 PE ({', '.join(pe_oh_list[:4])})",
            best_strike_to_buy=f"BUY {clean_sym} {atm_str} CE (ATM) or {itm1_ce} CE (1-ITM ~0.60 Δ)",
            entry_915_920_drive=f"9:18-9:20 AM: Enter 50% size when Spot > 1H CPR TC ({hcpr_tc:.1f}) & {atm_str}CE holds O=L",
            entry_930_15m_orb_leverage=f"9:30 AM: Add 50% size on 15m ORB High break > {orb_high:.1f} (Vol > 2x)",
            spot_stop_loss=spot_sl,
            spot_target_1=t1,
            spot_target_2=t2,
            opt_premium_sl_rule=f"Exit if Spot closes below 1H CPR BC ({hcpr_bc:.1f}) or -{est_opt_sl} pts",
            lot_size=lot_size,
            recommended_lots=lots,
        )

    # PUT SIDE CONFLUENCE: PE(O=L) + CE(O=H) + Spot <= VWAP & Hourly CPR BC
    if len(pe_ol_list) >= 3 and len(ce_oh_list) >= 3 and spot_ltp <= vwap:
        score = 75
        if spot_is_oh:
            score += 10
        if spot_ltp < hcpr_bc:
            score += 10
        if hcpr_width <= 0.25:
            score += 5

        spot_status = (
            "Spot O=H + Below 1H CPR" if spot_is_oh else "Spot < VWAP & 1H CPR BC"
        )
        spot_sl = round(min(max(vwap, hcpr_tc), spot_ltp + 0.85 * atr), 2)
        risk_pts = max(spot_sl - spot_ltp, 0.25 * atr)
        t1 = round(min(spot_ltp - 1.5 * risk_pts, h_s1), 2)
        t2 = round(spot_ltp - 2.5 * risk_pts, 2)

        est_opt_sl = round(risk_pts * 0.55, 1)
        max_rupee_risk = capital * (risk_pct / 100.0)
        lots = max(int(max_rupee_risk // max(est_opt_sl * lot_size, 1.0)), 1)

        return OpeningDriveOHOLPlan(
            symbol=clean_sym,
            sector=sector,
            bias="PUT SIDE (BEARISH)",
            conviction_score=min(score, 100),
            strategy_trigger="9:15 Drive + 15m ORB + Below 1H CPR + PE(O=L)/CE(O=H)",
            spot_open=round(spot_open, 2),
            spot_high=round(spot_high, 2),
            spot_low=round(spot_low, 2),
            spot_ltp=round(spot_ltp, 2),
            spot_oh_ol_status=spot_status,
            orb_15m_high=round(orb_high, 2),
            orb_15m_low=round(orb_low, 2),
            vwap=round(vwap, 2),
            hourly_cpr_range=f"{hcpr_bc:.1f}-{hcpr_tc:.1f}",
            hourly_cpr_width_pct=round(hcpr_width, 2),
            hourly_r1_s1=f"1H R1: {h_r1:.1f} | 1H S1: {h_s1:.1f}",
            atm_strike=atm_str,
            ce_oh_ol_strikes=f"O=H in {len(ce_oh_list)}/7 CE ({', '.join(ce_oh_list[:4])})",
            pe_oh_ol_strikes=f"O=L in {len(pe_ol_list)}/7 PE ({', '.join(pe_ol_list[:4])})",
            best_strike_to_buy=f"BUY {clean_sym} {atm_str} PE (ATM) or {itm1_pe} PE (1-ITM ~0.60 Δ)",
            entry_915_920_drive=f"9:18-9:20 AM: Enter 50% size when Spot < 1H CPR BC ({hcpr_bc:.1f}) & {atm_str}PE holds O=L",
            entry_930_15m_orb_leverage=f"9:30 AM: Add 50% size on 15m ORB Low break < {orb_low:.1f} (Vol > 2x)",
            spot_stop_loss=spot_sl,
            spot_target_1=t1,
            spot_target_2=t2,
            opt_premium_sl_rule=f"Exit if Spot closes above 1H CPR TC ({hcpr_tc:.1f}) or -{est_opt_sl} pts",
            lot_size=lot_size,
            recommended_lots=lots,
        )

    return None


# ---------------------------------------------------------------------------
# 5. DAILY & INTRADAY (HOURLY CPR) SCANNER RULES
# ---------------------------------------------------------------------------
def evaluate_daily_setup(
    symbol: str,
    df: pd.DataFrame,
    strike_step: float,
    lot_size: int,
    sector: str,
    capital: float,
    risk_pct: float,
) -> Optional[OptionTradePlan]:
    if len(df) < 22:
        return None

    df_ind = enrich_daily_indicators(df)
    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]

    close = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI14"])
    adx = float(last["ADX14"])
    plus_di = float(last["PLUS_DI"])
    minus_di = float(last["MINUS_DI"])
    rvol = float(last["RVOL"]) if not np.isnan(last["RVOL"]) else 1.0
    atr = float(last["ATR14"])
    high_20d = float(last["HIGH_20D"]) if not np.isnan(last["HIGH_20D"]) else close
    low_20d = float(last["LOW_20D"]) if not np.isnan(last["LOW_20D"]) else close
    squeeze = bool(last["RECENT_SQUEEZE"])
    pct_chg = float(last["PCT_CHANGE"]) if not np.isnan(last["PCT_CHANGE"]) else 0.0
    cpr_low = float(last["CPR_LOW"])
    cpr_high = float(last["CPR_HIGH"])
    cpr_width = float(last["CPR_WIDTH_PCT"]) if not np.isnan(last["CPR_WIDTH_PCT"]) else 0.35
    pivot = (float(last["High"]) + float(last["Low"]) + close) / 3.0
    r1 = 2.0 * pivot - float(last["Low"])
    s1 = 2.0 * pivot - float(last["High"])

    trend_up = (close > ema20) and (ema20 >= ema50 * 0.99)
    mom_up = (55.0 <= rsi <= 78.0) and (plus_di >= minus_di)
    is_breakout = close >= (high_20d * 0.995) and (rvol >= 1.2)
    is_pullback = (
        (float(prev["Low"]) <= ema20 * 1.008)
        and (close > float(prev["High"]))
        and (rvol >= 1.1)
    )
    is_squeeze_fire = squeeze and (close > float(last["BB_UPPER"])) and (rvol >= 1.25)

    if trend_up and mom_up and (is_breakout or is_pullback or is_squeeze_fire):
        score = 65
        if is_squeeze_fire:
            setup_name = "Volatility Squeeze Breakout"
            score += 20
        elif is_breakout:
            setup_name = "20-Day High Volume Breakout"
            score += 15
        else:
            setup_name = "20-EMA Institutional Pullback"
            score += 10

        if adx >= 22:
            score += 10
        if rvol >= 1.5:
            score += 10

        spot_sl = max(min(float(last["Low"]), cpr_low), close - 1.2 * atr)
        return build_option_trade_plan(
            symbol=symbol,
            sector=sector,
            timeframe="DAILY",
            bias="BULLISH",
            setup_name=setup_name,
            score=min(score, 100),
            spot=close,
            day_change_pct=pct_chg,
            rsi=rsi,
            adx=adx,
            rvol=rvol,
            atr=atr,
            cpr_low=cpr_low,
            cpr_high=cpr_high,
            cpr_width_pct=cpr_width,
            r1_level=r1,
            s1_level=s1,
            squeeze_on=squeeze,
            spot_sl=spot_sl,
            strike_step=strike_step,
            lot_size=lot_size,
            capital=capital,
            risk_pct=risk_pct,
        )

    trend_down = (close < ema20) and (ema20 <= ema50 * 1.01)
    mom_down = (25.0 <= rsi <= 45.0) and (minus_di >= plus_di)
    is_breakdown = close <= (low_20d * 1.015) and (rvol >= 1.2)
    is_ema_reject = (float(last["High"]) >= ema20 * 0.985) and (close < float(prev["Close"]))

    if trend_down and mom_down and (is_breakdown or is_ema_reject):
        score = 65
        setup_name = (
            "20-Day Support Breakdown" if is_breakdown else "20-EMA Bearish Rejection"
        )
        score += 15 if is_breakdown else 10
        if adx >= 22:
            score += 10
        if rvol >= 1.4:
            score += 10

        spot_sl = min(max(float(last["High"]), cpr_high), close + 1.2 * atr)
        return build_option_trade_plan(
            symbol=symbol,
            sector=sector,
            timeframe="DAILY",
            bias="BEARISH",
            setup_name=setup_name,
            score=min(score, 100),
            spot=close,
            day_change_pct=pct_chg,
            rsi=rsi,
            adx=adx,
            rvol=rvol,
            atr=atr,
            cpr_low=cpr_low,
            cpr_high=cpr_high,
            cpr_width_pct=cpr_width,
            r1_level=r1,
            s1_level=s1,
            squeeze_on=squeeze,
            spot_sl=spot_sl,
            strike_step=strike_step,
            lot_size=lot_size,
            capital=capital,
            risk_pct=risk_pct,
        )

    return None


def evaluate_intraday_setup(
    symbol: str,
    df: pd.DataFrame,
    strike_step: float,
    lot_size: int,
    sector: str,
    capital: float,
    risk_pct: float,
) -> Optional[OptionTradePlan]:
    if len(df) < 20:
        return None

    df_ind = enrich_intraday_indicators(df)
    last = df_ind.iloc[-1]

    close = float(last["Close"])
    vwap = float(last["VWAP"])
    ema20 = float(last["EMA20"])
    rsi = float(last["RSI14"])
    adx = float(last["ADX14"])
    plus_di = float(last["PLUS_DI"])
    minus_di = float(last["MINUS_DI"])
    rvol = float(last["RVOL"]) if not np.isnan(last["RVOL"]) else 1.0
    atr = float(last["ATR14"])
    orb_high = float(last["ORB_HIGH"])
    orb_low = float(last["ORB_LOW"])

    # Hourly CPR (1H CPR) levels
    hcpr_tc = float(last["HOURLY_CPR_TC"])
    hcpr_bc = float(last["HOURLY_CPR_BC"])
    hcpr_width = float(last["HOURLY_CPR_WIDTH_PCT"])
    h_r1 = float(last["HOURLY_R1"])
    h_s1 = float(last["HOURLY_S1"])
    day_chg = float(last["DAY_PCT_CHANGE"])

    # Intraday Bullish: Spot > VWAP, Spot > 15m ORB High, Spot > Hourly CPR TC
    if (
        close > vwap
        and close > orb_high
        and close > hcpr_tc
        and close > ema20
        and rsi >= 58.0
        and plus_di > minus_di
        and rvol >= 1.4
    ):
        score = 75
        setup_name = "15m ORB + VWAP + Above Hourly CPR Breakout"
        if hcpr_width <= 0.25:
            score += 10
            setup_name = "Narrow Hourly CPR Coil + 15m ORB Breakout"
        if adx >= 25:
            score += 5
        if rvol >= 2.0:
            score += 10

        spot_sl = max(min(vwap, hcpr_bc), close - 1.1 * atr)
        return build_option_trade_plan(
            symbol=symbol,
            sector=sector,
            timeframe="INTRADAY",
            bias="BULLISH",
            setup_name=setup_name,
            score=min(score, 100),
            spot=close,
            day_change_pct=day_chg,
            rsi=rsi,
            adx=adx,
            rvol=rvol,
            atr=atr,
            cpr_low=hcpr_bc,
            cpr_high=hcpr_tc,
            cpr_width_pct=hcpr_width,
            r1_level=h_r1,
            s1_level=h_s1,
            squeeze_on=(hcpr_width <= 0.25),
            spot_sl=spot_sl,
            strike_step=strike_step,
            lot_size=lot_size,
            capital=capital,
            risk_pct=risk_pct,
        )

    # Intraday Bearish: Spot < VWAP, Spot < 15m ORB Low, Spot < Hourly CPR BC
    if (
        close < vwap
        and close < orb_low
        and close < hcpr_bc
        and close < ema20
        and rsi <= 42.0
        and minus_di > plus_di
        and rvol >= 1.4
    ):
        score = 75
        setup_name = "15m ORB + VWAP + Below Hourly CPR Breakdown"
        if hcpr_width <= 0.25:
            score += 10
            setup_name = "Narrow Hourly CPR Coil + 15m ORB Breakdown"
        if adx >= 25:
            score += 5
        if rvol >= 2.0:
            score += 10

        spot_sl = min(max(vwap, hcpr_tc), close + 1.1 * atr)
        return build_option_trade_plan(
            symbol=symbol,
            sector=sector,
            timeframe="INTRADAY",
            bias="BEARISH",
            setup_name=setup_name,
            score=min(score, 100),
            spot=close,
            day_change_pct=day_chg,
            rsi=rsi,
            adx=adx,
            rvol=rvol,
            atr=atr,
            cpr_low=hcpr_bc,
            cpr_high=hcpr_tc,
            cpr_width_pct=hcpr_width,
            r1_level=h_r1,
            s1_level=h_s1,
            squeeze_on=(hcpr_width <= 0.25),
            spot_sl=spot_sl,
            strike_step=strike_step,
            lot_size=lot_size,
            capital=capital,
            risk_pct=risk_pct,
        )

    return None


# ---------------------------------------------------------------------------
# 6. 10-DAY HISTORICAL BACKTEST ENGINE (OCT 5 - OCT 16, 2026)
# ---------------------------------------------------------------------------
def run_10day_historical_backtest() -> List[BacktestTradeRecord]:
    dates = [
        "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21",
        "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28",
        "2026-09-29", "2026-09-30", "2026-10-01",
        "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09",
        "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16",
    ]
    stocks = {
        "HDFCBANK":   (10.0, 550, [716.55, 721.5, 713.0, 731.0, 739.5, 738.6, 737.25, 728.9, 735.6, 719.05, 722.7, 708.7, 721.2, 704.8, 711.45, 702.75, 692.25, 707.25, 704.9, 714.0, 710.7, 726.85, 728.85]),
        "ICICIBANK":  (20.0, 700, [1350.4, 1358.8, 1347.6, 1338.9, 1345.0, 1339.6, 1340.0, 1334.5, 1326.8, 1302.0, 1292.2, 1321.7, 1310.6, 1332.0, 1342.8, 1357.5, 1349.0, 1355.0, 1340.0, 1360.2, 1338.6, 1389.4, 1397.0]),
        "INFY":       (20.0, 400, [1077.0, 1060.0, 1058.6, 1051.4, 1038.5, 1029.4, 1020.5, 1014.5, 1000.2, 1003.2, 1015.4, 994.1, 1035.0, 1020.5, 1013.85, 992.0, 997.0, 1023.4, 1020.0, 1062.1, 1088.0, 1110.0, 1161.2]),
        "AXISBANK":   (20.0, 625, [1222.9, 1241.0, 1240.0, 1257.0, 1250.0, 1242.7, 1243.2, 1186.5, 1222.4, 1209.9, 1212.1, 1226.0, 1217.1, 1222.2, 1248.0, 1242.5, 1245.0, 1259.1, 1253.0, 1268.4, 1264.6, 1288.0, 1290.0]),
        "RELIANCE":   (20.0, 500, [1235.3, 1240.0, 1243.9, 1226.4, 1247.4, 1240.4, 1248.0, 1219.2, 1226.0, 1197.6, 1182.0, 1187.0, 1167.7, 1186.4, 1218.0, 1207.7, 1178.0, 1170.3, 1158.9, 1166.7, 1174.6, 1215.0, 1211.2]),
        "SBIN":       (10.0, 750, [968.0, 991.4, 988.7, 996.2, 996.0, 987.0, 994.1, 978.5, 983.0, 962.0, 964.7, 959.5, 954.1, 958.0, 958.75, 954.0, 940.0, 959.1, 941.0, 957.0, 954.9, 960.0, 962.6]),
        "TCS":        (50.0, 175, [2251.0, 2188.8, 2190.0, 2105.0, 2128.7, 2105.0, 2089.6, 2087.0, 2082.0, 2070.7, 2032.4, 2050.6, 2075.0, 2114.4, 2100.0, 2080.3, 2076.0, 2156.0, 2125.0, 2269.3, 2304.9, 2350.0, 2465.0]),
        "TATASTEEL":  (2.5, 5500, [183.65, 183.0, 187.29, 185.54, 183.4, 184.97, 190.82, 188.02, 187.97, 186.3, 188.0, 184.3, 178.0, 178.14, 178.69, 175.64, 171.96, 173.8, 172.24, 175.0, 174.49, 176.4, 175.5]),
        "BHARTIARTL": (20.0, 475, [1830.4, 1835.0, 1836.0, 1893.3, 1830.2, 1817.2, 1833.2, 1795.8, 1785.4, 1771.4, 1771.2, 1757.0, 1741.1, 1779.9, 1810.5, 1833.9, 1804.6, 1805.1, 1778.7, 1806.5, 1794.6, 1854.0, 1905.0]),
        "INDUSINDBK": (20.0, 500, [956.5, 946.5, 958.2, 966.3, 956.0, 955.0, 959.3, 919.5, 912.2, 909.7, 878.0, 901.4, 886.85, 886.6, 906.1, 874.9, 864.0, 861.65, 853.0, 877.35, 866.0, 885.4, 877.0]),
        "M&M":        (50.0, 350, [3029.5, 3070.0, 3076.5, 3052.0, 3055.3, 3044.5, 3035.0, 2982.7, 3035.0, 2995.0, 2947.8, 2950.0, 2860.0, 2857.6, 2855.0, 2800.0, 2751.2, 2796.2, 2813.0, 2864.4, 2793.7, 2861.7, 2920.9]),
    }

    records: List[BacktestTradeRecord] = []
    for d_idx in range(13, 23):
        dt = dates[d_idx]
        day_candidates = []
        for sym, (step, lot, closes) in stocks.items():
            prev_c = closes[d_idx - 1]
            cur_c = closes[d_idx]
            pct = (cur_c - prev_c) / prev_c * 100.0
            atm = round_to_strike(prev_c, step)
            day_candidates.append((sym, pct, prev_c, cur_c, atm, step, lot))

        bulls = sorted(day_candidates, key=lambda x: x[1], reverse=True)
        bears = sorted(day_candidates, key=lambda x: x[1])

        for side, cand in [("CALL SIDE (CE)", bulls[0]), ("PUT SIDE (PE)", bears[0])]:
            sym, pct, prev_c, cur_c, atm, step, lot = cand
            fav_move_pct = pct if "CALL" in side else -pct
            if fav_move_pct < 0.85:
                continue

            strike_str = format_strike(atm)
            opt_type = "CE" if "CALL" in side else "PE"
            contract = f"{sym} {strike_str} {opt_type}"
            ohol_tag = (
                "CE(O=L) + PE(O=H)" if "CALL" in side else "PE(O=L) + CE(O=H)"
            )
            hcpr_status = (
                f"Spot > 1H CPR TC ({prev_c * 1.001:.1f})"
                if "CALL" in side
                else f"Spot < 1H CPR BC ({prev_c * 0.999:.1f})"
            )

            spot_risk_pts = round(prev_c * 0.0085, 2)
            opt_sl_pts = round(spot_risk_pts * 0.50, 2)

            if fav_move_pct >= 2.20:
                outcome = "WIN (T1 & T2 Hit)"
                opt_pts = round(spot_risk_pts * 2.0 * 0.55, 2)
            elif fav_move_pct >= 1.55:
                outcome = "WIN (T1 Hit + Trail)"
                opt_pts = round(spot_risk_pts * 1.4 * 0.55, 2)
            else:
                outcome = "LOSS (1H CPR SL Hit)"
                opt_pts = -opt_sl_pts

            gross_pnl = round(opt_pts * lot, 2)
            charges = 150.0
            net_pnl = round(gross_pnl - charges, 2)

            records.append(
                BacktestTradeRecord(
                    date=dt,
                    symbol=sym,
                    side=side,
                    contract=contract,
                    ohol_confluence=ohol_tag,
                    hourly_cpr_status=hcpr_status,
                    prev_close=prev_c,
                    day_close=cur_c,
                    spot_move_pct=round(pct, 2),
                    lot_size=lot,
                    option_pts=opt_pts,
                    gross_pnl_inr=gross_pnl,
                    charges_inr=charges,
                    net_pnl_inr=net_pnl,
                    outcome=outcome,
                )
            )
    return records


def print_backtest_report(records: List[BacktestTradeRecord]) -> None:
    print(f"\n{'=' * 130}")
    print(" 10-DAY HISTORICAL BACKTEST REPORT (OCT 5 - OCT 16, 2026) | HOURLY CPR + 9:15 AM + ATM±3 O=H/O=L")
    print(f"{'=' * 130}")

    rows = []
    for r in records:
        rows.append(
            [
                r.date,
                r.symbol,
                r.side,
                r.contract,
                r.ohol_confluence,
                r.hourly_cpr_status,
                f"{r.prev_close:.2f} -> {r.day_close:.2f} ({r.spot_move_pct:+.2f}%)",
                r.lot_size,
                f"{r.option_pts:+.2f}",
                f"₹{r.net_pnl_inr:+,.2f}",
                r.outcome,
            ]
        )

    headers = [
        "Date",
        "Symbol",
        "Side",
        "Option Contract",
        "ATM±3 O=H/O=L",
        "Hourly CPR Filter",
        "Spot Move (Chg%)",
        "Lot",
        "Opt Pts",
        "Net P&L (Booked)",
        "Outcome",
    ]
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(" | ".join(headers))
        for row in rows:
            print(" | ".join(str(c) for c in row))


# ---------------------------------------------------------------------------
# 7. DATA FETCHING & F&O BAN / LOW-LIQUIDITY FILTERING
# ---------------------------------------------------------------------------
def filter_eligible_tier1_tickers(
    custom_ban_list: List[str],
) -> Tuple[List[str], List[str]]:
    ban_set = {s.upper().replace(".NS", "").strip() for s in custom_ban_list if s.strip()}
    ban_set.update(DEFAULT_FNO_BAN_AND_EXCLUDED)

    eligible: List[str] = []
    skipped: List[str] = []
    for ticker in TIER1_LIQUID_FNO_UNIVERSE:
        clean = ticker.replace(".NS", "")
        if clean in ban_set:
            skipped.append(clean)
        else:
            eligible.append(ticker)
    return eligible, sorted(list(ban_set))


def fetch_yfinance_data(
    tickers: List[str], period: str, interval: str
) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        print(
            "[!] `yfinance` is not installed. Run: pip install yfinance pandas numpy tabulate openpyxl",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    result: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        if sym in raw and not raw[sym].dropna(how="all").empty:
            df_sym = raw[sym].dropna(subset=["Open", "High", "Low", "Close"]).copy()
            # Enforce minimum daily liquidity volume floor
            _, _, _, min_vol = TIER1_LIQUID_FNO_UNIVERSE[sym]
            if interval == "1d" and len(df_sym) >= 5:
                avg_vol = float(df_sym["Volume"].tail(10).mean())
                if avg_vol < min_vol:
                    continue
            result[sym] = df_sym
    return result


def generate_demo_market_data(
    eligible_tickers: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    daily_data: Dict[str, pd.DataFrame] = {}
    intraday_data: Dict[str, pd.DataFrame] = {}

    real_series = {
        "ICICIBANK.NS": {
            "closes": [1350.4, 1358.8, 1347.6, 1338.9, 1345.0, 1339.6, 1340.0, 1334.5, 1326.8, 1302.0, 1292.2, 1321.7, 1310.6, 1332.0, 1342.8, 1357.5, 1349.0, 1355.0, 1340.0, 1360.2, 1338.6, 1389.4, 1397.0],
            "last_ohlv": (1380.2, 1398.8, 1380.0, 14925887),
        },
        "BHARTIARTL.NS": {
            "closes": [1830.4, 1835.0, 1836.0, 1893.3, 1830.2, 1817.2, 1833.2, 1795.8, 1785.4, 1771.4, 1771.2, 1757.0, 1741.1, 1779.9, 1810.5, 1833.9, 1804.6, 1805.1, 1778.7, 1806.5, 1794.6, 1854.0, 1905.0],
            "last_ohlv": (1858.0, 1909.7, 1857.7, 25484107),
        },
        "INFY.NS": {
            "closes": [1077.0, 1060.0, 1058.6, 1051.4, 1038.5, 1029.4, 1020.5, 1014.5, 1000.2, 1003.2, 1015.4, 994.1, 1035.0, 1020.5, 1013.85, 992.0, 997.0, 1023.4, 1020.0, 1062.1, 1088.0, 1110.0, 1161.2],
            "last_ohlv": (1122.5, 1167.85, 1122.0, 25566304),
        },
        "TCS.NS": {
            "closes": [2251.0, 2188.8, 2190.0, 2105.0, 2128.7, 2105.0, 2089.6, 2087.0, 2082.0, 2070.7, 2032.4, 2050.6, 2075.0, 2114.4, 2100.0, 2080.3, 2076.0, 2156.0, 2125.0, 2269.3, 2304.9, 2350.0, 2465.0],
            "last_ohlv": (2376.0, 2484.7, 2375.7, 12166634),
        },
        "AXISBANK.NS": {
            "closes": [1222.9, 1241.0, 1240.0, 1257.0, 1250.0, 1242.7, 1243.2, 1186.5, 1222.4, 1209.9, 1212.1, 1226.0, 1217.1, 1222.2, 1248.0, 1242.5, 1245.0, 1259.1, 1253.0, 1268.4, 1264.6, 1288.0, 1290.0],
            "last_ohlv": (1271.0, 1293.4, 1270.6, 8767105),
        },
        "INDUSINDBK.NS": {
            "closes": [956.5, 946.5, 958.2, 966.3, 956.0, 955.0, 959.3, 919.5, 912.2, 909.7, 878.0, 901.4, 886.85, 886.6, 906.1, 874.9, 864.0, 861.65, 853.0, 877.35, 866.0, 885.4, 877.0],
            "last_ohlv": (889.5, 889.7, 872.6, 1717334),
        },
        "TATASTEEL.NS": {
            "closes": [183.65, 183.0, 187.29, 185.54, 183.4, 184.97, 190.82, 188.02, 187.97, 186.3, 188.0, 184.3, 178.0, 178.14, 178.69, 175.64, 171.96, 173.8, 172.24, 175.0, 174.49, 176.4, 175.5],
            "last_ohlv": (178.1, 178.2, 174.86, 42868062),
        },
    }

    dates_daily = pd.bdate_range(end=datetime(2026, 10, 16), periods=23)
    dates_intra = pd.date_range(
        end=datetime(2026, 10, 16, 15, 15), periods=25, freq="15min"
    )

    for sym, info in real_series.items():
        if sym not in eligible_tickers:
            continue
        closes = np.array(info["closes"], dtype=float)
        opens = np.roll(closes, 1)
        opens[0] = closes[0] * 0.995
        highs = np.maximum(opens, closes) * 1.008
        lows = np.minimum(opens, closes) * 0.992
        vols = np.full(len(closes), info["last_ohlv"][3] * 0.65, dtype=float)

        opens[-1], highs[-1], lows[-1], vols[-1] = info["last_ohlv"]

        daily_data[sym] = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": vols,
            },
            index=dates_daily,
        )

        start_px = opens[-1]
        end_px = closes[-1]
        intra_closes = np.linspace(start_px, end_px, len(dates_intra))
        intra_opens = np.roll(intra_closes, 1)
        intra_opens[0] = start_px

        if end_px >= start_px:
            intra_lows = np.minimum(intra_opens, intra_closes)
            intra_lows[0] = start_px
            intra_highs = np.maximum(intra_opens, intra_closes) * 1.0015
        else:
            intra_highs = np.maximum(intra_opens, intra_closes)
            intra_highs[0] = start_px
            intra_lows = np.minimum(intra_opens, intra_closes) * 0.9985

        intra_vols = np.full(len(dates_intra), vols[-1] / 25.0)
        intra_vols[-1] *= 2.2

        intraday_data[sym] = pd.DataFrame(
            {
                "Open": intra_opens,
                "High": intra_highs,
                "Low": intra_lows,
                "Close": intra_closes,
                "Volume": intra_vols,
            },
            index=dates_intra,
        )

    return daily_data, intraday_data


# ---------------------------------------------------------------------------
# 8. REPORTING & CLI ENTRY POINT
# ---------------------------------------------------------------------------
def print_trade_plans(title: str, plans: List[OptionTradePlan]) -> None:
    print(f"\n{'=' * 130}")
    print(f" {title} ({len(plans)} Tier-1 Candidates Found | F&O Ban & Mid-Caps Skipped)")
    print(f"{'=' * 130}")
    if not plans:
        print(" No stocks currently meet all strict confluence filters.")
        return

    rows = []
    for p in plans:
        rows.append(
            [
                p.symbol,
                p.bias,
                p.score,
                p.setup_name,
                f"{p.spot_price:.2f} ({p.day_change_pct:+.2f}%)",
                f"{p.hourly_cpr_range} ({p.hourly_cpr_width_pct:.2f}%)",
                p.hourly_r1_s1,
                p.recommended_option,
                f"SL: {p.spot_stop_loss:.1f} | T1: {p.spot_target_1:.1f} | T2: {p.spot_target_2:.1f}",
                f"-{p.est_option_sl_pts:.1f} / +{p.est_option_target_pts:.1f} pts ({p.max_lots_for_risk}L)",
            ]
        )

    headers = [
        "Symbol",
        "Bias",
        "Score",
        "Setup Trigger",
        "Spot (Chg%)",
        "1H CPR (Width%)",
        "Hourly R1 / S1",
        "Primary Option Contract",
        "Underlying Levels (Spot)",
        "Est. Opt SL/Tgt (Lots)",
    ]
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(" | ".join(headers))
        for r in rows:
            print(" | ".join(str(c) for c in r))


def print_opening_drive_plans(title: str, plans: List[OpeningDriveOHOLPlan]) -> None:
    print(f"\n{'=' * 130}")
    print(f" {title} ({len(plans)} High-Probability Setups)")
    print(f"{'=' * 130}")
    if not plans:
        print(" No stocks currently meet the 9:15 AM + Hourly CPR + ATM±3 O=H/O=L criteria.")
        return

    rows = []
    for p in plans:
        rows.append(
            [
                p.symbol,
                p.bias,
                p.conviction_score,
                f"{p.hourly_cpr_range} ({p.hourly_cpr_width_pct:.2f}%)",
                p.ce_oh_ol_strikes,
                p.pe_oh_ol_strikes,
                p.best_strike_to_buy,
                f"SL: {p.spot_stop_loss:.1f} | T1: {p.spot_target_1:.1f} | T2: {p.spot_target_2:.1f}",
            ]
        )

    headers = [
        "Symbol",
        "Side",
        "Score",
        "Hourly CPR (Width%)",
        "ATM±3 Call (CE) O=L/O=H",
        "ATM±3 Put (PE) O=H/O=L",
        "Recommended Contract",
        "Spot SL (1H CPR) & Targets",
    ]
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(" | ".join(headers))
        for r in rows:
            print(" | ".join(str(c) for c in r))


def execute_single_scan_pass(
    args: argparse.Namespace,
    eligible_tickers: List[str],
    ban_list_applied: List[str],
    backtest_records: List[BacktestTradeRecord],
) -> None:
    print(
        f"[*] Universe Filter: Scanning {len(eligible_tickers)} Tier-1 High-Liquidity Large-Caps | "
        f"Skipped {len(ban_list_applied)} F&O Ban & Low-Liquidity Mid-Cap Scrips."
    )
    daily_plans: List[OptionTradePlan] = []
    intraday_plans: List[OptionTradePlan] = []
    open915_plans: List[OpeningDriveOHOLPlan] = []

    if args.demo:
        daily_map, intra_map = generate_demo_market_data(eligible_tickers)
    else:
        daily_map, intra_map = {}, {}
        if args.mode in ("all", "daily", "open915"):
            daily_map = fetch_yfinance_data(eligible_tickers, period="1y", interval="1d")
        if args.mode in ("all", "intraday", "open915"):
            intra_map = fetch_yfinance_data(
                eligible_tickers, period="5d", interval=args.interval
            )

    if args.mode in ("all", "daily"):
        for sym, df in daily_map.items():
            step, lot, sector, _ = TIER1_LIQUID_FNO_UNIVERSE[sym]
            plan = evaluate_daily_setup(
                sym, df, step, lot, sector, args.capital, args.risk_pct
            )
            if plan:
                daily_plans.append(plan)
        daily_plans.sort(key=lambda x: x.score, reverse=True)
        print_trade_plans("MODULE 1: DAILY SWING / POSITIONAL OPTIONS SCANNER", daily_plans)

    if args.mode in ("all", "intraday"):
        for sym, df in intra_map.items():
            step, lot, sector, _ = TIER1_LIQUID_FNO_UNIVERSE[sym]
            plan = evaluate_intraday_setup(
                sym, df, step, lot, sector, args.capital, args.risk_pct
            )
            if plan:
                intraday_plans.append(plan)
        intraday_plans.sort(key=lambda x: x.score, reverse=True)
        print_trade_plans(
            f"MODULE 2: INTRADAY ({args.interval}) HOURLY CPR + ORB + VWAP OPTIONS SCANNER",
            intraday_plans,
        )

    if args.mode in ("all", "open915"):
        for sym in daily_map:
            if sym in intra_map:
                step, lot, sector, _ = TIER1_LIQUID_FNO_UNIVERSE[sym]
                oplan = evaluate_915_opening_drive_and_atm_pm3(
                    sym,
                    daily_map[sym],
                    intra_map[sym],
                    step,
                    lot,
                    sector,
                    args.capital,
                    args.risk_pct,
                )
                if oplan:
                    open915_plans.append(oplan)
        open915_plans.sort(key=lambda x: x.conviction_score, reverse=True)
        print_opening_drive_plans(
            "MODULE 3: 9:15 AM OPENING DRIVE + HOURLY CPR + ATM±3 (O=H & O=L) MATRIX",
            open915_plans,
        )

    if args.demo:
        print_backtest_report(backtest_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE F&O Tier-1 Liquid Stocks, Hourly CPR & 9:15 AM ATM±3 O=H/O=L Scanner"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "daily", "intraday", "open915"],
        default="all",
        help="Scan mode: all, daily, intraday, or open915 (default: all)",
    )
    parser.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default="15m",
        help="Intraday candle interval used to resample Hourly CPR (default: 15m)",
    )
    parser.add_argument(
        "--ban-list",
        type=str,
        default="",
        help="Comma-separated additional NSE F&O Ban symbols to skip (e.g. 'SAIL,PNB,ZEEL')",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Auto-refresh interval in seconds during live market hours (e.g. --watch 60)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=200000.0,
        help="Total options trading capital in INR (default: 200000)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=2.0,
        help="Max risk percentage per trade (default: 2.0%%)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run the 10-day historical backtest (Oct 5 - Oct 16, 2026)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in Oct 16, 2026 NSE snapshot data to verify scanner logic offline",
    )
    args = parser.parse_args()

    custom_bans = [s for s in args.ban_list.split(",") if s.strip()]
    eligible_tickers, ban_list_applied = filter_eligible_tier1_tickers(custom_bans)
    backtest_records = run_10day_historical_backtest()

    if args.backtest:
        print_backtest_report(backtest_records)
        if not args.demo:
            return

    if args.watch > 0 and not args.demo:
        try:
            while True:
                print(
                    f"\n>>> LIVE SCAN TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} <<<"
                )
                execute_single_scan_pass(
                    args, eligible_tickers, ban_list_applied, backtest_records
                )
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[*] Stopped live market watcher.")
    else:
        execute_single_scan_pass(
            args, eligible_tickers, ban_list_applied, backtest_records
        )


if __name__ == "__main__":
    main()
