"""
Vector store abstraction over ChromaDB (local) or Pinecone (cloud).

SQLite fix for Python 3.9 / older systems: pysqlite3-binary ships its own
sqlite3 >= 3.35.0 which ChromaDB requires. Must be patched before import.

Collections:
  - "trade_history"  — closed trade records + outcomes
  - "lessons"        — Auditor-generated lessons from past trades
  - "market_cycles"  — Historical regime descriptions seeded by bootstrap_memory.py
"""
from __future__ import annotations

import os
import sys
import logging
import hashlib
from typing import Optional

# --- SQLite3 version fix (required for ChromaDB on Python 3.9 / older systems) ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # pysqlite3-binary not installed; hope system sqlite3 is >= 3.35.0
# ---------------------------------------------------------------------------------

import chromadb
from chromadb.config import Settings

from src.models import RAGDocument

logger = logging.getLogger(__name__)

COLLECTIONS = ["trade_history", "lessons", "market_cycles"]


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma_db") -> None:
        self._persist_dir = persist_dir
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}
        self._init_collections()

    def _init_collections(self) -> None:
        for name in COLLECTIONS:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info("VectorStore: %d collections ready", len(self._collections))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, collection: str, doc: RAGDocument) -> None:
        """Insert or update a document. doc.embedding must be set."""
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")
        if doc.embedding is None:
            raise ValueError("doc.embedding must be set before upserting")

        col = self._collections[collection]
        col.upsert(
            ids=[doc.doc_id],
            embeddings=[doc.embedding],
            documents=[doc.content],
            metadatas=[doc.metadata],
        )

    def upsert_batch(self, collection: str, docs: list[RAGDocument]) -> None:
        if not docs:
            return
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"doc {doc.doc_id} has no embedding")

        col = self._collections[collection]
        col.upsert(
            ids=[d.doc_id for d in docs],
            embeddings=[d.embedding for d in docs],
            documents=[d.content for d in docs],
            metadatas=[d.metadata for d in docs],
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[RAGDocument]:
        """Return top-n documents by cosine similarity."""
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")

        col = self._collections[collection]
        kwargs: dict = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        docs: list[RAGDocument] = []

        if not results["ids"] or not results["ids"][0]:
            return docs

        for i, doc_id in enumerate(results["ids"][0]):
            docs.append(
                RAGDocument(
                    doc_id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    embedding=results["embeddings"][0][i] if results.get("embeddings") else None,
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def collection_count(self, collection: str) -> int:
        return self._collections[collection].count()

    @staticmethod
    def make_doc_id(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
