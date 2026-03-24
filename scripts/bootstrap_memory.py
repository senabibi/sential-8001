"""
Bootstrap SENTINEL-8004's RAG memory with synthetic experience.

Pipeline:
  1. Load BTC/USD and ETH/USD hourly CSVs from data/historical/
  2. Compute technical indicators (RSI, EMA, ATR, Volume Z-score, etc.)
  3. Detect significant market events (drops, spikes, vol anomalies, RSI extremes)
  4. For each event: send 48-candle context + 24-candle outcome to GPT-4o Auditor
  5. Store the resulting "lesson" in the RAG (ChromaDB "lessons" collection)
  6. Also chunk the full OHLCV history into "market_cycles" for regime-matching

Usage (after running prepare_data.py):
    python scripts/bootstrap_memory.py
    python scripts/bootstrap_memory.py --skip-lessons      # just index market cycles, no LLM
    python scripts/bootstrap_memory.py --max-events 50     # limit LLM calls (saves cost)
    python scripts/bootstrap_memory.py --since 2022-01-01  # only recent data
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import pandas as pd

from src.llm_client import get_client, chat_with_fallback, BOOTSTRAP_MODEL

from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.processing.cleaner import (
    load_ohlcv_df,
    detect_significant_events,
    get_context_window,
    pair_from_filename,
)
from src.processing.indicators import classify_market_regime
from src.processing.embedder import market_data_to_chunks, lesson_to_document
from src.models import OHLCV, MarketData, ExecutionLayer

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bootstrap_memory")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"

# Primary files to process (hourly gives the best event resolution)
DATA_FILES = [
    ("XBTUSD_60.csv", "BTC/USD"),
    ("ETHUSD_60.csv", "ETH/USD"),
]

# -------------------------------------------------------------------------
# Auditor prompt (event-based, indicator-rich)
# -------------------------------------------------------------------------

AUDITOR_PROMPT = """You are a senior quantitative trading analyst reviewing a historical market event.

=== MARKET CONTEXT (48 candles before the event) ===
Pair: {pair}
Timeframe: 1H candles

At the event candle ({event_dt} UTC):
  - Price:          {event_close:.2f} USD
  - Single-candle change: {event_chg:+.2f}%
  - RSI(14):        {rsi:.1f}  [{rsi_label}]
  - EMA20:          {ema20:.2f}  |  EMA50: {ema50:.2f}  ({ema_gap:+.2f}% gap)
  - ATR(14):        {atr:.2f}  (volatility unit)
  - Volume Z-score: {vol_z:+.2f}  [{vol_label}]
  - Bollinger %B:   {bb:.2f}  [0=lower band, 1=upper band]
  - Market regime:  {regime}

24-hour pre-event trend: {pre_trend:+.2f}%
48-hour pre-event high/low: [{pre_low:.2f}, {pre_high:.2f}]

=== OUTCOME (next 24 candles after event) ===
  - Price 24h later: {outcome_close:.2f}  ({outcome_chg:+.2f}% from event)
  - Max recovery:    {outcome_high:.2f}   Max further drop: {outcome_low:.2f}
  - Outcome trend:   {outcome_label}

=== YOUR TASK ===
A trading agent was considering {direction} {pair} at the event candle.
Analyse what happened and write a concise "Lesson Note" (2-4 sentences) that:
1. Names the warning sign(s) visible BEFORE the event
2. States what the agent should have done at the event candle
3. Names ONE specific risk parameter adjustment (e.g. "tighten stop_loss_pct from 2% to 1%")

Output ONLY the lesson text — no JSON, no markdown headers.
"""


def _build_lesson_prompt(
    pair: str,
    event_row: pd.Series,
    context_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
) -> str:
    event_dt = event_row["datetime"].strftime("%Y-%m-%d %H:%M") if "datetime" in event_row else "unknown"
    event_close = float(event_row["close"])
    event_chg = float(event_row.get("price_chg_pct", 0))

    rsi = float(event_row.get("rsi14", 50))
    rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

    ema20 = float(event_row.get("ema20", event_close))
    ema50 = float(event_row.get("ema50", event_close))
    ema_gap = (ema20 - ema50) / ema50 * 100 if ema50 != 0 else 0.0

    atr = float(event_row.get("atr14", 0))
    vol_z = float(event_row.get("vol_zscore", 0))
    vol_label = "very high" if vol_z > 2 else "high" if vol_z > 1 else "normal" if abs(vol_z) < 1 else "low"
    bb = float(event_row.get("bb_pct", 0.5))
    trend_ema = float(event_row.get("trend_ema20_50", 0))

    regime = classify_market_regime(rsi, event_chg, vol_z, bb, trend_ema)

    if not context_df.empty:
        pre_trend = (context_df["close"].iloc[-1] - context_df["close"].iloc[0]) / context_df["close"].iloc[0] * 100
        pre_high = float(context_df["high"].max())
        pre_low = float(context_df["low"].min())
    else:
        pre_trend = pre_high = pre_low = 0.0

    if not outcome_df.empty:
        outcome_close = float(outcome_df["close"].iloc[-1])
        outcome_high = float(outcome_df["high"].max())
        outcome_low = float(outcome_df["low"].min())
        outcome_chg = (outcome_close - event_close) / event_close * 100 if event_close != 0 else 0
        outcome_label = "recovered" if outcome_chg > 2 else "continued_down" if outcome_chg < -2 else "flat"
    else:
        outcome_close = event_close
        outcome_high = event_close
        outcome_low = event_close
        outcome_chg = 0.0
        outcome_label = "unknown"

    # Agent direction based on event type
    event_type = str(event_row.get("event_type", ""))
    direction = "selling" if "drop" in event_type or "overbought" in event_type else "buying"

    return AUDITOR_PROMPT.format(
        pair=pair,
        event_dt=event_dt,
        event_close=event_close,
        event_chg=event_chg,
        rsi=rsi,
        rsi_label=rsi_label,
        ema20=ema20,
        ema50=ema50,
        ema_gap=ema_gap,
        atr=atr,
        vol_z=vol_z,
        vol_label=vol_label,
        bb=bb,
        regime=regime,
        pre_trend=pre_trend,
        pre_high=pre_high,
        pre_low=pre_low,
        outcome_close=outcome_close,
        outcome_chg=outcome_chg,
        outcome_high=outcome_high,
        outcome_low=outcome_low,
        outcome_label=outcome_label,
        direction=direction,
    )


def generate_lesson(pair: str, event_dt: str, prompt: str) -> str:
    try:
        return chat_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            model=BOOTSTRAP_MODEL,   # llama-3.1-8b-instant: 500k tokens/day
            max_tokens=300,
            temperature=0.3,
        )
    except Exception as e:
        logger.error("LLM call failed for %s %s: %s", pair, event_dt, e)
        return ""


# -------------------------------------------------------------------------
# Phase 1: Index market cycles (no LLM needed)
# -------------------------------------------------------------------------

def index_market_cycles(
    df: pd.DataFrame,
    pair: str,
    retriever: Retriever,
    vector_store: VectorStore,
    chunk_size: int = 48,
    stride: int = 24,
) -> int:
    """Chunk the OHLCV history into narrative paragraphs and embed them."""
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
    market_data = MarketData(
        pair=pair,
        candles=candles,
        current_price=candles[-1].close,
        source=ExecutionLayer.KRAKEN,
    )

    from src.processing.embedder import market_data_to_chunks
    chunks = market_data_to_chunks(market_data, chunk_size=chunk_size, stride=stride)
    logger.info("Indexing %d market cycle chunks for %s...", len(chunks), pair)

    batch_size = 20
    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            embeddings = retriever.embed_batch([d.content for d in batch])
            for doc, emb in zip(batch, embeddings):
                doc.embedding = emb
            vector_store.upsert_batch("market_cycles", batch)
            stored += len(batch)
        except Exception as e:
            logger.error("Embedding batch %d failed: %s", i // batch_size, e)
        logger.info("  Indexed %d / %d chunks", min(stored, len(chunks)), len(chunks))
        time.sleep(0.5)  # Rate limit courtesy

    return stored


# -------------------------------------------------------------------------
# Phase 2: Event-based lesson generation
# -------------------------------------------------------------------------

def generate_event_lessons(
    df: pd.DataFrame,
    pair: str,
    retriever: Retriever,
    vector_store: VectorStore,
    max_events: int = 200,
    min_drop_pct: float = -3.0,
) -> int:
    """Detect events, generate GPT-4o lessons, embed and store in RAG."""
    events_df = detect_significant_events(
        df,
        price_drop_threshold=min_drop_pct,
        price_spike_threshold=3.0,
        volume_zscore_threshold=2.5,
        rsi_overbought=72.0,
        rsi_oversold=28.0,
    )

    if events_df.empty:
        logger.warning("No events detected for %s", pair)
        return 0

    # Prioritise: worst drops > volume spikes > RSI extremes > price spikes
    def _priority(row: pd.Series) -> float:
        et = str(row.get("event_type", ""))
        score = 0.0
        if "price_drop" in et:
            score += abs(row.get("price_chg_pct", 0)) * 2
        if "volume_sell_spike" in et or "volume_buy_spike" in et:
            score += abs(row.get("vol_zscore", 0)) * 1.5
        if "rsi_oversold" in et or "rsi_overbought" in et:
            score += 5.0
        if "price_spike" in et:
            score += abs(row.get("price_chg_pct", 0))
        return score

    events_df = events_df.copy()
    events_df["_priority"] = events_df.apply(_priority, axis=1)
    events_df = events_df.sort_values("_priority", ascending=False).reset_index(drop=True)

    total = min(max_events, len(events_df))
    logger.info("Generating lessons for top %d events (of %d) in %s", total, len(events_df), pair)

    stored = 0
    for i, event_row in events_df.head(total).iterrows():
        df_idx = int(event_row.get("df_index", i))
        context_df, outcome_df = get_context_window(df, df_idx, before=48, after=24)

        event_dt = (
            event_row["datetime"].strftime("%Y-%m-%d %H:%M")
            if "datetime" in event_row
            else str(event_row.get("timestamp", ""))
        )

        prompt = _build_lesson_prompt(pair, event_row, context_df, outcome_df)
        lesson = generate_lesson(pair, event_dt, prompt)

        if not lesson:
            continue

        doc = lesson_to_document(lesson)
        doc.metadata.update({
            "pair": pair,
            "event_type": str(event_row.get("event_type", "")),
            "event_dt": event_dt,
            "price_chg_pct": round(float(event_row.get("price_chg_pct", 0)), 4),
            "rsi14": round(float(event_row.get("rsi14", 50)), 2),
            "vol_zscore": round(float(event_row.get("vol_zscore", 0)), 2),
            "regime": classify_market_regime(
                float(event_row.get("rsi14", 50)),
                float(event_row.get("price_chg_pct", 0)),
                float(event_row.get("vol_zscore", 0)),
                float(event_row.get("bb_pct", 0.5)),
                float(event_row.get("trend_ema20_50", 0)),
            ),
        })

        try:
            doc.embedding = retriever.embed(lesson)
            vector_store.upsert("lessons", doc)
            stored += 1
            logger.info(
                "[%d/%d] %s %s %s | %.2f%% | rsi=%.1f | stored lesson",
                stored, total, pair, event_dt,
                event_row.get("event_type", ""), event_row.get("price_chg_pct", 0),
                event_row.get("rsi14", 50),
            )
        except Exception as e:
            logger.error("Failed to embed/store lesson: %s", e)

        time.sleep(2.5)  # Groq free tier: ~30 req/min, 2.5s keeps us under

    return stored


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap SENTINEL-8004 RAG memory")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with historical CSVs")
    parser.add_argument("--since", default="2022-01-01", help="Start date YYYY-MM-DD (default: 2022-01-01)")
    parser.add_argument("--until", default=None, help="End date YYYY-MM-DD (default: all)")
    parser.add_argument("--max-events", type=int, default=200, help="Max lessons to generate per pair (default: 200)")
    parser.add_argument("--min-drop", type=float, default=-3.0, help="Min %% drop to classify as event (default: -3)")
    parser.add_argument("--skip-lessons", action="store_true", help="Skip LLM lesson generation (no API calls)")
    parser.add_argument("--skip-cycles", action="store_true", help="Skip market cycle indexing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    until = datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.until else None

    if not args.skip_lessons:
        try:
            from src.llm_client import get_groq_api_key
            get_groq_api_key()
        except RuntimeError as e:
            logger.error("%s\nUse --skip-lessons to skip LLM generation.", e)
            sys.exit(1)

    # client no longer needed — generate_lesson uses chat_with_fallback directly
    vector_store = VectorStore()
    retriever = Retriever(vector_store)

    total_cycles = 0
    total_lessons = 0

    for filename, pair in DATA_FILES:
        path = data_dir / filename
        if not path.exists():
            logger.warning("File not found: %s — run prepare_data.py first", path)
            continue

        logger.info("=" * 60)
        logger.info("Processing %s → %s (since %s)", filename, pair, args.since)

        df = load_ohlcv_df(path, since=since, until=until)
        if df.empty:
            logger.warning("No data after date filtering — skipping")
            continue

        logger.info("  Loaded %d candles [%s → %s]",
                    len(df),
                    df["datetime"].iloc[0].strftime("%Y-%m-%d"),
                    df["datetime"].iloc[-1].strftime("%Y-%m-%d"))

        # Phase 1: Market cycles
        if not args.skip_cycles:
            n = index_market_cycles(df, pair, retriever, vector_store)
            total_cycles += n
            logger.info("  Indexed %d market cycle chunks for %s", n, pair)

        # Phase 2: Event lessons
        if not args.skip_lessons:
            n = generate_event_lessons(
                df, pair, retriever, vector_store,
                max_events=args.max_events,
                min_drop_pct=args.min_drop,
            )
            total_lessons += n
            logger.info("  Generated %d lessons for %s", n, pair)

    # Final stats
    logger.info("=" * 60)
    logger.info("Bootstrap complete.")
    logger.info("  Market cycles stored: %d", total_cycles)
    logger.info("  Lessons stored:       %d", total_lessons)
    logger.info("  RAG collections:")
    logger.info("    market_cycles: %d documents", vector_store.collection_count("market_cycles"))
    logger.info("    lessons:       %d documents", vector_store.collection_count("lessons"))
    logger.info("    trade_history: %d documents", vector_store.collection_count("trade_history"))
    logger.info("")
    logger.info("The agent now has synthetic experience from %s onward.", args.since)
    logger.info("Next step: run the simulator to let it self-correct further.")


if __name__ == "__main__":
    main()
