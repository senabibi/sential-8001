"""
Kraken CLI execution worker.

STUB — to be implemented by teammate.

This file defines the class signature and documents what each method
must do so the orchestrator works as-is once filled in.

Implementation notes for teammate:
  - Use subprocess to call the Kraken CLI binary (or MCP server mode)
  - CLI binary path from env: KRAKEN_CLI_PATH (default: "kraken")
  - API keys from env: KRAKEN_API_KEY, KRAKEN_API_SECRET
  - All paper/sandbox orders use --sandbox flag on the CLI
  - Commands reference: see README.md "Kraken CLI" section
"""
from __future__ import annotations

import os
import uuid
import time
import logging

from src.core.base_worker import BaseWorker
from src.models import (
    Direction, ExecutionLayer, MarketData, OHLCV,
    Order, ExecutionResult, PortfolioState, Position,
)

logger = logging.getLogger(__name__)


class KrakenWorker(BaseWorker):
    """
    Wraps Kraken CLI (subprocess + optional MCP server).

    TODO (teammate): implement the four abstract methods.
    Until then, the paper-trading simulator injects market data directly
    and the orchestrator falls back gracefully when this raises NotImplementedError.
    """

    def __init__(self) -> None:
        self._cli_path = os.getenv("KRAKEN_CLI_PATH", "kraken")
        self._sandbox = os.getenv("TRADE_MODE", "paper") == "paper"

    @property
    def name(self) -> str:
        return "kraken"

    # ------------------------------------------------------------------
    # TODO: implement below
    # ------------------------------------------------------------------

    async def get_market_data(self, pair: str, num_candles: int = 100) -> MarketData:
        """
        Call:
            kraken market ticker --pair {kraken_pair}
            kraken market ohlc   --pair {kraken_pair} --interval 1

        Parse JSON output and return MarketData.
        """
        raise NotImplementedError(
            "KrakenWorker.get_market_data not yet implemented. "
            "Run in simulation mode or implement the Kraken CLI subprocess calls."
        )

    async def get_portfolio(self) -> PortfolioState:
        """
        Call:
            kraken account balance
            kraken account positions

        Parse and return PortfolioState.
        """
        raise NotImplementedError("KrakenWorker.get_portfolio not yet implemented.")

    async def execute_order(self, order: Order) -> ExecutionResult:
        """
        Call (example market buy):
            kraken [--sandbox] order add \\
                --pair {kraken_pair} \\
                --type buy \\
                --ordertype market \\
                --volume {volume}

        Map ExecutionLayer.KRAKEN, parse txid from response.
        """
        raise NotImplementedError("KrakenWorker.execute_order not yet implemented.")

    async def close_position(self, pair: str, sandbox: bool = True) -> ExecutionResult:
        """Close all open positions for pair at market."""
        raise NotImplementedError("KrakenWorker.close_position not yet implemented.")
