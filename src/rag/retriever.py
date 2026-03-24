"""
RAG retriever — embeds queries and fetches relevant context from the vector store.

Embeddings: sentence-transformers (local, free, no API key needed).
Model: all-MiniLM-L6-v2 — 384-dim, fast, good semantic similarity.
Downloaded automatically on first use (~90 MB, cached in ~/.cache/huggingface/).
"""
from __future__ import annotations

import logging
from typing import Optional

from src.rag.vector_store import VectorStore
from src.models import MarketData, ClosedTrade, RAGDocument

logger = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model():
    """Lazy-load the embedding model (downloaded once, then cached)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model all-MiniLM-L6-v2 (first-time download ~90MB)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model ready.")
    return _embed_model


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        model = _get_embed_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = _get_embed_model()
        return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_similar_market_conditions(
        self, market_data: MarketData, n: int = 3
    ) -> list[RAGDocument]:
        summary = self._summarise_market(market_data)
        embedding = self.embed(summary)
        return self._store.query("market_cycles", embedding, n_results=n)

    def get_similar_trades(
        self, pair: str, direction: str, n: int = 3
    ) -> list[RAGDocument]:
        text = f"Trade on {pair}, direction {direction}"
        embedding = self.embed(text)
        return self._store.query(
            "trade_history", embedding, n_results=n,
            where={"pair": pair},
        )

    def get_lessons(self, context: str, n: int = 3) -> list[RAGDocument]:
        embedding = self.embed(context)
        return self._store.query("lessons", embedding, n_results=n)

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    def store_trade(self, trade: ClosedTrade) -> None:
        content = (
            f"Trade {trade.trade_id}: {trade.direction.value} {trade.pair} | "
            f"Entry {trade.entry_price:.4f} → Exit {trade.exit_price:.4f} | "
            f"PnL {trade.pnl_usd:+.2f} USD ({trade.pnl_pct:+.2f}%)"
        )
        doc = RAGDocument(
            doc_id=VectorStore.make_doc_id(content),
            content=content,
            metadata={
                "pair": trade.pair,
                "direction": trade.direction.value,
                "pnl_usd": trade.pnl_usd,
                "timestamp": trade.exit_timestamp,
                "layer": trade.execution_layer.value,
            },
        )
        doc.embedding = self.embed(content)
        self._store.upsert("trade_history", doc)

    def store_lesson(self, lesson: str, metadata: Optional[dict] = None) -> None:
        doc = RAGDocument(
            doc_id=VectorStore.make_doc_id(lesson),
            content=lesson,
            metadata=metadata or {},
        )
        doc.embedding = self.embed(lesson)
        self._store.upsert("lessons", doc)

    def store_market_cycle(self, description: str, metadata: Optional[dict] = None) -> None:
        doc = RAGDocument(
            doc_id=VectorStore.make_doc_id(description),
            content=description,
            metadata=metadata or {},
        )
        doc.embedding = self.embed(description)
        self._store.upsert("market_cycles", doc)

    # ------------------------------------------------------------------
    # Format RAG context for LLM prompt
    # ------------------------------------------------------------------

    def format_context(self, docs: list[RAGDocument], header: str = "Relevant context") -> str:
        if not docs:
            return ""
        parts = [f"--- {header} ---"]
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] {doc.content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_market(md: MarketData) -> str:
        if not md.candles:
            return f"{md.pair} current price {md.current_price}"
        candles = md.candles[-20:]
        prices = [c.close for c in candles]
        high = max(c.high for c in candles)
        low = min(c.low for c in candles)
        change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100 if prices[0] != 0 else 0
        return (
            f"{md.pair}: current {md.current_price:.4f}, "
            f"20-candle change {change_pct:+.2f}%, "
            f"range [{low:.4f}, {high:.4f}]"
        )
