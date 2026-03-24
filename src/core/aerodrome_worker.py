"""
Aerodrome Finance execution worker (Base L2).

STUB — to be implemented by teammate.

Implementation notes for teammate:
  - Use web3.py to interact with Base L2 (RPC from env: BASE_RPC_URL)
  - Private key from env: AGENT_WALLET_PRIVATE_KEY
  - TradeIntents must be EIP-712 signed via src/onchain/signing.py before submission
  - All DeFi orders go through the Surge Risk Router (src/onchain/risk_router.py)
  - Pool analytics can be fetched from Aerodrome subgraph or direct contract calls
  - chain_id = 8453 (mainnet) or 84532 (Base Sepolia testnet)
"""
from __future__ import annotations

import logging

from src.core.base_worker import BaseWorker
from src.models import (
    ExecutionLayer, MarketData, Order, ExecutionResult, PortfolioState,
)

logger = logging.getLogger(__name__)


class AerodromeWorker(BaseWorker):
    """
    On-chain swap + LP worker via Aerodrome Finance on Base L2.

    TODO (teammate): implement the four abstract methods.
    """

    def __init__(self) -> None:
        pass  # TODO: initialise web3 provider, wallet, Risk Router contract

    @property
    def name(self) -> str:
        return "aerodrome"

    async def get_market_data(self, pair: str, num_candles: int = 100) -> MarketData:
        """
        Fetch price + pool data from Aerodrome subgraph or on-chain oracle.
        OHLCV can be approximated from block-by-block price queries or a 3rd-party API.
        """
        raise NotImplementedError("AerodromeWorker.get_market_data not yet implemented.")

    async def get_portfolio(self) -> PortfolioState:
        """
        Query ERC-20 token balances + open LP positions on Base L2.
        """
        raise NotImplementedError("AerodromeWorker.get_portfolio not yet implemented.")

    async def execute_order(self, order: Order) -> ExecutionResult:
        """
        1. Build TradeIntent from order
        2. EIP-712 sign it via src.onchain.signing.sign_eip712()
        3. Submit to Surge Risk Router via src.onchain.risk_router.submit()
        4. Wait for tx confirmation, return ExecutionResult with tx_hash
        """
        raise NotImplementedError("AerodromeWorker.execute_order not yet implemented.")

    async def close_position(self, pair: str, sandbox: bool = True) -> ExecutionResult:
        """Remove LP position or submit a reverse swap to close."""
        raise NotImplementedError("AerodromeWorker.close_position not yet implemented.")
