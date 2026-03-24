"""
Wallet utilities — EIP-155 chain binding + EIP-1271 smart contract wallet support.
STUB — to be implemented by teammate.
"""
from __future__ import annotations


def get_wallet_address(private_key: str) -> str:
    """Derive EOA address from private key."""
    raise NotImplementedError


async def is_contract_wallet(address: str, rpc_url: str) -> bool:
    """Return True if address is a smart contract (EIP-1271 wallet e.g. Safe)."""
    raise NotImplementedError


async def verify_eip1271(
    message_hash: bytes,
    signature: bytes,
    wallet_address: str,
    rpc_url: str,
) -> bool:
    """
    Call isValidSignature(bytes32, bytes) on a smart contract wallet.
    EIP-1271 magic value: 0x1626ba7e
    """
    raise NotImplementedError
