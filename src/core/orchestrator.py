"""
SENTINEL-8004 Orchestrator — main event loop.

Loop: Observe → Hypothesize → Gate → Execute → Audit → (repeat)

Usage:
    python -m src.core.orchestrator --mode paper
    python -m src.core.orchestrator --mode live
"""
from __future__ import annotations

import asyncio
import argparse
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.models import (
    Order, ClosedTrade, Direction, ExecutionLayer, PortfolioState,
    TradeSignal, TradeMode,
)
from src.agents.strategy_agent import StrategyAgent
from src.agents.risk_manager import RiskManager
from src.agents.auditor_agent import AuditorAgent
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.core.base_worker import BaseWorker

# On-chain layer — stubs until teammate implements
try:
    from src.onchain.reputation import post_reputation_update
    from src.onchain.validator import post_validation_artifact
    _ONCHAIN_AVAILABLE = True
except ImportError:
    _ONCHAIN_AVAILABLE = False

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel.orchestrator")

# Pairs to trade (can be overridden via env)
_DEFAULT_PAIRS = ["BTC/USD", "ETH/USD"]

# Seconds between full observation cycles
_CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL_SECONDS", "60"))


class Orchestrator:
    """
    Main trading loop. Composes all agents and workers.

    Pass execution workers via constructor so the orchestrator
    works identically in paper mode, simulation, and live mode.
    """

    def __init__(
        self,
        workers: list[BaseWorker],
        mode: TradeMode = TradeMode.PAPER,
        pairs: Optional[list[str]] = None,
        cycle_interval: int = _CYCLE_INTERVAL,
    ) -> None:
        self._workers = {w.name: w for w in workers}
        self._mode = mode
        self._pairs = pairs or _DEFAULT_PAIRS
        self._cycle_interval = cycle_interval
        self._sandbox = mode == TradeMode.PAPER

        # Shared infrastructure
        self._vector_store = VectorStore()
        self._retriever = Retriever(self._vector_store)
        self._risk_manager = RiskManager()
        self._strategy_agent = StrategyAgent(self._retriever)
        self._auditor_agent = AuditorAgent(self._risk_manager, self._retriever)

        # In-memory state for open positions (worker owns canonical state)
        self._open_orders: dict[str, dict] = {}   # trade_id → {order, signal, entry}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("SENTINEL-8004 starting | mode=%s | pairs=%s", self._mode.value, self._pairs)
        for worker in self._workers.values():
            await worker.connect()
        try:
            await self._loop()
        finally:
            for worker in self._workers.values():
                await worker.disconnect()

    async def _loop(self) -> None:
        while True:
            cycle_start = time.time()
            try:
                await self._cycle()
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
            elapsed = time.time() - cycle_start
            sleep_for = max(0.0, self._cycle_interval - elapsed)
            logger.debug("Cycle done in %.1fs, sleeping %.1fs", elapsed, sleep_for)
            await asyncio.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Single trading cycle
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        for pair in self._pairs:
            for worker_name, worker in self._workers.items():
                await self._process_pair(pair, worker)

    async def _process_pair(self, pair: str, worker: BaseWorker) -> None:
        # 1. OBSERVE
        try:
            market_data = await worker.get_market_data(pair)
            portfolio = await worker.get_portfolio()
        except NotImplementedError:
            logger.warning("Worker '%s' not yet implemented — skipping %s", worker.name, pair)
            return
        except Exception as e:
            logger.error("Failed to fetch data for %s via %s: %s", pair, worker.name, e)
            return

        # 2. HYPOTHESIZE — ask the Strategy Agent
        signal: TradeSignal = self._strategy_agent.generate_signal(market_data)

        # 3. GATE — Risk Manager
        decision = self._risk_manager.evaluate(signal, portfolio)
        if not decision.approved:
            logger.info("VETO [%s/%s]: %s", pair, worker.name, decision.reason)
            return

        # Apply any size adjustment from risk manager
        effective_size = decision.adjusted_size_pct or signal.size_pct

        # 4. EXECUTE
        trade_id = str(uuid.uuid4())[:8]
        order = Order(
            pair=pair,
            direction=signal.direction,
            size_pct=effective_size,
            execution_layer=ExecutionLayer(worker.name),
            sandbox=self._sandbox,
            trade_id=trade_id,
            stop_loss_pct=self._risk_manager.load_policy().get("stop_loss_pct"),
            take_profit_pct=self._risk_manager.load_policy().get("take_profit_pct"),
        )

        try:
            result = await worker.execute_order(order)
        except NotImplementedError:
            logger.warning("Worker '%s' execute_order not implemented", worker.name)
            return
        except Exception as e:
            logger.error("Execution error [%s]: %s", trade_id, e)
            return

        if not result.success:
            logger.error("Order failed [%s]: %s", trade_id, result.error)
            return

        logger.info(
            "EXECUTED [%s] %s %s @ %.4f | size=%.4f | layer=%s",
            trade_id, signal.direction.value, pair,
            result.executed_price, result.executed_size, worker.name,
        )

        # Track open position for later close + audit
        self._open_orders[result.trade_id] = {
            "order": order,
            "signal": signal,
            "result": result,
            "entry_price": result.executed_price,
            "entry_timestamp": result.timestamp,
        }

    # ------------------------------------------------------------------
    # Called externally when a position closes (e.g. stop-loss triggered)
    # ------------------------------------------------------------------

    async def on_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        exit_timestamp: int,
        worker_name: str,
    ) -> None:
        ctx = self._open_orders.pop(trade_id, None)
        if not ctx:
            logger.warning("on_trade_closed: unknown trade_id %s", trade_id)
            return

        order: Order = ctx["order"]
        signal: TradeSignal = ctx["signal"]
        result = ctx["result"]
        entry_price: float = ctx["entry_price"]

        if signal.direction == Direction.BUY:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_usd = pnl_pct / 100 * (result.executed_size * entry_price)

        closed = ClosedTrade(
            trade_id=trade_id,
            pair=order.pair,
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=result.executed_size,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            execution_layer=result.execution_layer,
            entry_timestamp=ctx["entry_timestamp"],
            exit_timestamp=exit_timestamp,
            signal=signal,
            order=order,
            result=result,
        )

        # 5. AUDIT (self-correction)
        audit_report = self._auditor_agent.audit(closed)

        # 6. ON-CHAIN trust signals (if layer available)
        if _ONCHAIN_AVAILABLE:
            try:
                await post_reputation_update(closed.pnl_usd)
                if audit_report.config_updates:
                    await post_validation_artifact(audit_report)
            except Exception as e:
                logger.warning("On-chain post failed: %s", e)

        logger.info(
            "AUDIT [%s] %s | PnL %.2f USD | lesson stored=%s",
            trade_id, audit_report.outcome, audit_report.pnl_usd,
            audit_report.rag_lesson_stored,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main(mode: str, pairs: Optional[list[str]] = None) -> None:
    trade_mode = TradeMode(mode)

    # Import workers (teammate implementations — fall back to stubs gracefully)
    from src.core.kraken_worker import KrakenWorker
    from src.core.aerodrome_worker import AerodromeWorker

    workers: list[BaseWorker] = [KrakenWorker()]
    if os.getenv("ENABLE_DEFI", "false").lower() == "true":
        workers.append(AerodromeWorker())

    orchestrator = Orchestrator(workers=workers, mode=trade_mode, pairs=pairs)
    await orchestrator.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL-8004 Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Pairs to trade, e.g. BTC/USD ETH/USD",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.mode, args.pairs))
