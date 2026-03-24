"""
Embedder — converts raw data (market snapshots, trade records, lessons)
into RAGDocument chunks ready for the vector store.
"""
from __future__ import annotations

import time
import logging
from typing import Optional

from src.models import MarketData, ClosedTrade, RAGDocument
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def market_data_to_chunks(
    market_data: MarketData,
    chunk_size: int = 48,
    stride: int = 24,
) -> list[RAGDocument]:
    """
    Slice OHLCV history into overlapping narrative chunks for the RAG.
    Each chunk describes a window of candles as a human-readable paragraph.
    """
    candles = market_data.candles
    chunks: list[RAGDocument] = []

    for start in range(0, len(candles) - chunk_size + 1, stride):
        window = candles[start : start + chunk_size]
        if len(window) < 2:
            continue

        opens  = [c.open   for c in window]
        closes = [c.close  for c in window]
        highs  = [c.high   for c in window]
        lows   = [c.low    for c in window]
        volumes = [c.volume for c in window]

        pct_change    = ((closes[-1] - opens[0]) / opens[0]) * 100 if opens[0] != 0 else 0
        max_drawdown  = ((min(lows) - opens[0]) / opens[0]) * 100 if opens[0] != 0 else 0
        avg_volume    = sum(volumes) / len(volumes)
        vol_range     = max(volumes) / avg_volume if avg_volume > 0 else 1.0  # max vol spike ratio

        if pct_change > 1:
            direction = "bullish"
        elif pct_change < -1:
            direction = "bearish"
        else:
            direction = "sideways"

        ts_start = window[0].timestamp
        ts_end   = window[-1].timestamp

        content = (
            f"{market_data.pair} market from ts={ts_start} to ts={ts_end}: "
            f"{direction} phase, {chunk_size}-candle change {pct_change:+.2f}%, "
            f"range [{min(lows):.2f}, {max(highs):.2f}], "
            f"max intra-window drawdown {max_drawdown:.2f}%, "
            f"avg volume {avg_volume:.2f}, max volume spike {vol_range:.1f}x."
        )

        chunks.append(
            RAGDocument(
                doc_id=VectorStore.make_doc_id(content),
                content=content,
                metadata={
                    "pair":        market_data.pair,
                    "ts_start":    ts_start,
                    "ts_end":      ts_end,
                    "pct_change":  round(pct_change, 4),
                    "direction":   direction,
                    "max_drawdown":round(max_drawdown, 4),
                    "source":      market_data.source.value,
                },
            )
        )

    logger.debug(
        "Chunked %d candles → %d RAG documents for %s",
        len(candles), len(chunks), market_data.pair,
    )
    return chunks


def trade_to_document(trade: ClosedTrade) -> RAGDocument:
    """Convert a closed trade into a single RAGDocument."""
    outcome = "profitable" if trade.pnl_usd > 0 else "unprofitable"
    content = (
        f"[Trade {trade.trade_id}] {trade.direction.value.upper()} {trade.pair} | "
        f"{outcome}: entry {trade.entry_price:.4f} → exit {trade.exit_price:.4f} | "
        f"PnL {trade.pnl_usd:+.2f} USD ({trade.pnl_pct:+.2f}%) | "
        f"layer: {trade.execution_layer.value}"
    )
    return RAGDocument(
        doc_id=VectorStore.make_doc_id(f"{trade.trade_id}:{trade.exit_timestamp}"),
        content=content,
        metadata={
            "trade_id":   trade.trade_id,
            "pair":       trade.pair,
            "direction":  trade.direction.value,
            "pnl_usd":    trade.pnl_usd,
            "pnl_pct":    trade.pnl_pct,
            "entry_ts":   trade.entry_timestamp,
            "exit_ts":    trade.exit_timestamp,
            "layer":      trade.execution_layer.value,
        },
    )


def lesson_to_document(lesson: str, metadata: Optional[dict] = None) -> RAGDocument:
    """Wrap an Auditor-generated lesson text into a RAGDocument."""
    ts = int(time.time())
    base_meta: dict = {"type": "lesson", "timestamp": ts}
    if metadata:
        base_meta.update(metadata)
    return RAGDocument(
        doc_id=VectorStore.make_doc_id(lesson),
        content=lesson,
        metadata=base_meta,
    )
