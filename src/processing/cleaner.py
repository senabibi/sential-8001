"""
Data cleaning — loads Kraken OHLCV CSV files and normalises them.

Supported formats:
  A) No header:  timestamp, open, high, low, close, volume, trades
     Files:      XBTUSD_60.csv, ETHUSD_60.csv, XBTUSD_1440.csv, ETHUSD_1440.csv, etc.

  B) With header: timestamp, open, high, low, close, volume, trades
     Files:      BTCUSD_Daily_OHLC.csv, BTCUSD_1.csv

Interval suffix → minutes:
  _1     → 1 min
  _5     → 5 min
  _15    → 15 min
  _30    → 30 min
  _60    → 60 min (1H)
  _240   → 4H
  _720   → 12H
  _1440  → 1D
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models import OHLCV, MarketData, ExecutionLayer
from src.processing.indicators import add_all_indicators

logger = logging.getLogger(__name__)

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_ohlcv_df(
    csv_path: str | Path,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load a Kraken OHLCV CSV into a clean, indicator-enriched DataFrame.

    Args:
        csv_path: Path to .csv file
        since:    Only keep candles >= this UTC datetime
        until:    Only keep candles <= this UTC datetime

    Returns:
        DataFrame with columns:
            timestamp (int, unix epoch), open, high, low, close, volume, trades,
            datetime (UTC), rsi14, ema20, ema50, ema200, atr14, vol_zscore,
            bb_pct, price_chg_pct, trend_ema20_50
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {path}")

    # Auto-detect header
    first_line = path.open().readline().strip()
    has_header = first_line.lower().startswith("timestamp") or first_line[0].isalpha()

    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else _COLUMNS,
    )

    # Ensure expected columns exist
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path.name}")

    df = df.dropna(subset=["timestamp", "open", "close"])
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Date filtering
    if since is not None:
        since_ts = int(since.replace(tzinfo=timezone.utc).timestamp())
        df = df[df["timestamp"] >= since_ts]
    if until is not None:
        until_ts = int(until.replace(tzinfo=timezone.utc).timestamp())
        df = df[df["timestamp"] <= until_ts]

    df = df.reset_index(drop=True)

    if df.empty:
        logger.warning("No rows after date filtering: %s", path.name)
        return df

    # Human-readable datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Add technical indicators
    df = add_all_indicators(df)

    logger.info(
        "Loaded %s: %d candles [%s → %s]",
        path.name,
        len(df),
        df["datetime"].iloc[0].strftime("%Y-%m-%d"),
        df["datetime"].iloc[-1].strftime("%Y-%m-%d"),
    )
    return df


def load_ohlcv_market_data(
    csv_path: str | Path,
    pair: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    max_candles: Optional[int] = None,
) -> MarketData:
    """Load CSV → MarketData (for RAG / strategy agent)."""
    df = load_ohlcv_df(csv_path, since=since, until=until)

    if max_candles and len(df) > max_candles:
        df = df.tail(max_candles).reset_index(drop=True)

    candles = [
        OHLCV(
            timestamp=int(row.timestamp),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for row in df.itertuples()
    ]

    current_price = candles[-1].close if candles else 0.0
    vol_24h = df["volume"].tail(24).sum() if len(df) >= 24 else df["volume"].sum()

    return MarketData(
        pair=pair,
        candles=candles,
        current_price=current_price,
        volume_24h=float(vol_24h),
        source=ExecutionLayer.KRAKEN,
    )


# ---------------------------------------------------------------------------
# Event detection (for bootstrap_memory.py)
# ---------------------------------------------------------------------------

def detect_significant_events(
    df: pd.DataFrame,
    price_drop_threshold: float = -3.0,    # % single-candle drop
    price_spike_threshold: float = 3.0,    # % single-candle spike
    volume_zscore_threshold: float = 2.5,  # std devs above mean
    rsi_overbought: float = 72.0,
    rsi_oversold: float = 28.0,
) -> pd.DataFrame:
    """
    Detect significant market events from indicator-enriched DataFrame.

    Returns a filtered DataFrame of event rows with an added 'event_type' column.
    Multiple event types can fire on the same candle (stored as comma-separated string).
    """
    if df.empty or "rsi14" not in df.columns:
        raise ValueError("DataFrame must have indicators — call load_ohlcv_df() first")

    events = []
    for idx, row in df.iterrows():
        types = []
        chg = row.get("price_chg_pct", 0.0)
        vol_z = row.get("vol_zscore", 0.0)
        rsi = row.get("rsi14", 50.0)

        if chg <= price_drop_threshold:
            types.append("price_drop")
        if chg >= price_spike_threshold:
            types.append("price_spike")
        if vol_z >= volume_zscore_threshold and chg < 0:
            types.append("volume_sell_spike")
        if vol_z >= volume_zscore_threshold and chg > 0:
            types.append("volume_buy_spike")
        if rsi >= rsi_overbought:
            types.append("rsi_overbought")
        if rsi <= rsi_oversold:
            types.append("rsi_oversold")

        if types:
            events.append({**row.to_dict(), "event_type": ",".join(types), "df_index": idx})

    result = pd.DataFrame(events)
    logger.info(
        "Detected %d significant events in %d candles", len(result), len(df)
    )
    return result


def get_context_window(
    df: pd.DataFrame,
    event_idx: int,
    before: int = 48,    # candles of context before event
    after: int = 24,     # candles after (outcome window)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (context_df, outcome_df) around an event index in the full df.
    context_df: `before` candles ending at event_idx (inclusive)
    outcome_df: `after` candles AFTER event_idx
    """
    start = max(0, event_idx - before + 1)
    context = df.iloc[start : event_idx + 1]
    outcome = df.iloc[event_idx + 1 : event_idx + 1 + after]
    return context, outcome


# ---------------------------------------------------------------------------
# Pair name normalisation
# ---------------------------------------------------------------------------

_PAIR_MAP = {
    "XBTUSD": "BTC/USD",
    "XBTUSD1": "BTC/USD",
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
    "ETHUSD1": "ETH/USD",
    "XBTUSDC": "BTC/USDC",
    "ETHUSDC": "ETH/USDC",
}


def normalise_pair(raw: str) -> str:
    key = raw.upper().split("_")[0]
    return _PAIR_MAP.get(key, raw)


def pair_from_filename(filename: str) -> str:
    stem = Path(filename).stem          # e.g. "XBTUSD_60"
    raw = stem.split("_")[0]            # "XBTUSD"
    return normalise_pair(raw)
