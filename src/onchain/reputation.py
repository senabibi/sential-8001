"""
ERC-8004 Reputation Registry — post PnL signals after trade close.
STUB — to be implemented by teammate.
"""
from __future__ import annotations


async def post_reputation_update(pnl_usd: float) -> str:
    """
    Post a PnL delta to the ERC-8004 Reputation Registry.
    Returns the transaction hash.

    TODO (teammate):
        - Call ReputationRegistry.updateReputation(agentId, pnl_usd_scaled)
        - Return tx hash
    """
    raise NotImplementedError("post_reputation_update not yet implemented")
