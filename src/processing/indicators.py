"""
Technical indicators used for event detection and RAG context building.
Pure Python + pandas — no TA-lib dependency.

Indicators:
  - RSI(n)            — Relative Strength Index
  - EMA(n)            — Exponential Moving Average
  - ATR(n)            — Average True Range (volatility)
  - volume_zscore(n)  — Volume Z-score (how unusual is this volume?)
  - price_change_pct  — Single-candle % change
  - bb_pct            — Bollinger Band %B (where price sits in the band)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    vol_mean = volume.rolling(period, min_periods=1).mean()
    vol_std = volume.rolling(period, min_periods=1).std().replace(0, np.nan)
    return ((volume - vol_mean) / vol_std).fillna(0.0)


def compute_bb_pct(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger Band %B: 0 = at lower band, 1 = at upper band, 0.5 = at SMA."""
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std().replace(0, np.nan)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_pct = (close - lower) / (upper - lower)
    return bb_pct.clip(0.0, 1.0).fillna(0.5)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicators to a DataFrame with columns:
        timestamp, open, high, low, close, volume
    Returns a new DataFrame with additional columns.
    """
    df = df.copy()
    df["rsi14"] = compute_rsi(df["close"], 14)
    df["ema20"] = compute_ema(df["close"], 20)
    df["ema50"] = compute_ema(df["close"], 50)
    df["ema200"] = compute_ema(df["close"], 200)
    df["atr14"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["vol_zscore"] = compute_volume_zscore(df["volume"], 20)
    df["bb_pct"] = compute_bb_pct(df["close"], 20)
    df["price_chg_pct"] = df["close"].pct_change() * 100
    df["trend_ema20_50"] = (df["ema20"] - df["ema50"]) / df["ema50"] * 100  # % gap
    return df


def classify_market_regime(
    rsi: float,
    price_chg_pct: float,
    vol_zscore: float,
    bb_pct: float,
    trend_ema: float,
) -> str:
    """
    Simple rule-based regime classification for RAG metadata.
    Returns one of: bullish_breakout, bearish_breakdown, overbought, oversold,
                    volume_spike_up, volume_spike_down, sideways, normal
    """
    if vol_zscore > 2.0 and price_chg_pct > 3.0:
        return "volume_spike_up"
    if vol_zscore > 2.0 and price_chg_pct < -3.0:
        return "volume_spike_down"
    if rsi > 75 and bb_pct > 0.9:
        return "overbought"
    if rsi < 25 and bb_pct < 0.1:
        return "oversold"
    if price_chg_pct > 5.0 and trend_ema > 1.0:
        return "bullish_breakout"
    if price_chg_pct < -5.0 and trend_ema < -1.0:
        return "bearish_breakdown"
    if abs(price_chg_pct) < 0.5 and abs(trend_ema) < 0.5:
        return "sideways"
    return "normal"
