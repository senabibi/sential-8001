"""
Abstract base class for execution workers.

Your teammate implements:
  - KrakenWorker(BaseWorker)      in kraken_worker.py
  - AerodromeWorker(BaseWorker)   in aerodrome_worker.py

The Orchestrator only calls these abstract methods, so both layers
are interchangeable from the orchestrator's perspective.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import MarketData, Order, ExecutionResult, PortfolioState


class BaseWorker(ABC):
    """Execution worker interface. One implementation per execution layer."""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_market_data(self, pair: str, num_candles: int = 100) -> MarketData:
        """
        Fetch current market data for a pair.

        Args:
            pair: Trading pair, e.g. "BTC/USD" or "WETH/USDC"
            num_candles: Number of OHLCV candles to return (most-recent last)

        Returns:
            MarketData with current price, bid/ask, and OHLCV history
        """
        ...

    @abstractmethod
    async def get_portfolio(self) -> PortfolioState:
        """Return current balances and open positions."""
        ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute_order(self, order: Order) -> ExecutionResult:
        """
        Submit an order.

        Args:
            order: Fully specified Order (pair, direction, size_pct, etc.)
                   order.sandbox == True → paper/sandbox mode
                   order.sandbox == False → live execution

        Returns:
            ExecutionResult — always returned even on failure (check .success)
        """
        ...

    @abstractmethod
    async def close_position(self, pair: str, sandbox: bool = True) -> ExecutionResult:
        """Close all open positions for a pair at market price."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Optional: establish any persistent connections / authenticate."""
        pass

    async def disconnect(self) -> None:
        """Optional: clean up connections on shutdown."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'kraken' or 'aerodrome'."""
        ...
