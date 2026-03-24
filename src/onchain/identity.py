"""
ERC-8004 Agent Identity — mint and register on Base L2.
STUB — to be implemented by teammate.
"""
from __future__ import annotations


async def mint_agent_identity(
    name: str,
    capabilities: list[str],
    metadata_uri: str,
    private_key: str,
) -> int:
    """
    Mint an ERC-721 Agent Identity NFT on Base L2.
    Returns the token ID.

    TODO (teammate):
        - Upload config/agent_registration.json to IPFS → get metadata_uri
        - Call IdentityRegistry.registerAgent(name, capabilities, metadata_uri)
        - Return emitted token ID
    """
    raise NotImplementedError("mint_agent_identity not yet implemented")


async def get_agent_id(wallet_address: str) -> int | None:
    """Return the agent's ERC-8004 token ID, or None if not registered."""
    raise NotImplementedError("get_agent_id not yet implemented")
