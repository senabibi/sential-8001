"""
Surge Risk Router — submit signed TradeIntents for on-chain validation + DeFi execution.
STUB — to be implemented by teammate.
"""
from __future__ import annotations

from src.models import TradeIntent


async def submit(signed_intent: TradeIntent, signature: str) -> str:
    """
    Submit a signed TradeIntent to the Surge Risk Router contract.
    Returns the transaction hash.

    The Risk Router enforces:
      - Max position size per agent
      - Max leverage
      - Whitelisted markets
      - Daily loss limit

    TODO (teammate):
        - Call RiskRouter.submitTradeIntent(intent_struct, signature)
        - Parse emitted events to confirm execution
        - Return tx hash
    """
    raise NotImplementedError("risk_router.submit not yet implemented")
