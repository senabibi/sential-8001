"""
Shared data models for SENTINEL-8004.
All inter-module data flows through these Pydantic models.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ExecutionLayer(str, Enum):
    KRAKEN = "kraken"
    AERODROME = "aerodrome"


class TradeMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class OHLCV(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketData(BaseModel):
    pair: str
    candles: list[OHLCV]           # Most-recent last
    current_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None
    source: ExecutionLayer = ExecutionLayer.KRAKEN


# ---------------------------------------------------------------------------
# Trade signal (output of StrategyAgent)
# ---------------------------------------------------------------------------

class TradeSignal(BaseModel):
    pair: str
    direction: Direction
    size_pct: float                # % of available capital to allocate
    confidence: float              # 0.0 – 1.0
    reasoning: str
    rag_context_used: bool = False
    strategy_model: str = "claude-sonnet-4-6"
    timestamp: int = Field(default_factory=lambda: int(time.time()))


# ---------------------------------------------------------------------------
# Risk decision (output of RiskManager)
# ---------------------------------------------------------------------------

class RiskDecision(BaseModel):
    approved: bool
    reason: str
    adjusted_size_pct: Optional[float] = None   # If capped, what it was adjusted to
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Order (passed to execution worker)
# ---------------------------------------------------------------------------

class Order(BaseModel):
    pair: str
    direction: Direction
    size_pct: float
    order_type: str = "market"     # "market" | "limit"
    limit_price: Optional[float] = None
    execution_layer: ExecutionLayer = ExecutionLayer.KRAKEN
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    sandbox: bool = True
    trade_id: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))


# ---------------------------------------------------------------------------
# Execution result (returned by execution worker)
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    success: bool
    trade_id: str
    pair: str
    direction: Direction
    executed_price: float
    executed_size: float           # In base asset units
    execution_layer: ExecutionLayer
    tx_hash: Optional[str] = None  # DeFi only
    error: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))


# ---------------------------------------------------------------------------
# Closed trade (input to AuditorAgent)
# ---------------------------------------------------------------------------

class ClosedTrade(BaseModel):
    trade_id: str
    pair: str
    direction: Direction
    entry_price: float
    exit_price: float
    size: float
    pnl_usd: float
    pnl_pct: float
    execution_layer: ExecutionLayer
    entry_timestamp: int
    exit_timestamp: int
    signal: Optional[TradeSignal] = None
    order: Optional[Order] = None
    result: Optional[ExecutionResult] = None


# ---------------------------------------------------------------------------
# Audit report (output of AuditorAgent)
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    param: str
    old_value: float
    new_value: float
    reason: str


class AuditReport(BaseModel):
    trade_id: str
    outcome: str                   # "profit" | "loss" | "break_even"
    pnl_usd: float
    root_cause: str
    lesson: str
    config_updates: list[ConfigUpdate] = Field(default_factory=list)
    rag_lesson_stored: bool = False
    validation_artifact_hash: Optional[str] = None
    auditor_model: str = "gpt-4o"
    timestamp: int = Field(default_factory=lambda: int(time.time()))


# ---------------------------------------------------------------------------
# RAG document
# ---------------------------------------------------------------------------

class RAGDocument(BaseModel):
    doc_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# EIP-712 TradeIntent (for on-chain signing — populated by onchain layer)
# ---------------------------------------------------------------------------

class TradeIntent(BaseModel):
    agent_id: int = 0
    pair: str
    direction: str
    amount: int                    # In smallest unit (wei for DeFi, satoshi-equivalent for CeFi)
    dex: str                       # "kraken" | "aerodrome"
    chain_id: int = 8453
    nonce: int = 0
    deadline: int = Field(default_factory=lambda: int(time.time()) + 300)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

class Position(BaseModel):
    pair: str
    direction: Direction
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    execution_layer: ExecutionLayer


class PortfolioState(BaseModel):
    total_balance_usd: float
    available_balance_usd: float
    open_positions: list[Position] = Field(default_factory=list)
    daily_pnl_usd: float = 0.0
    daily_trade_count: int = 0
    last_trade_timestamp: int = 0
