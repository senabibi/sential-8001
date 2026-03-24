"""
EIP-712 typed data signing for TradeIntents.
STUB — to be implemented by teammate using web3.py / eth-account.

References:
  - https://eips.ethereum.org/EIPS/eip-712
  - eth_account.structured_data.sign_typed_data_message_with_signable_message
"""
from __future__ import annotations

from src.models import TradeIntent

# EIP-712 domain + type definitions (for teammate reference)
DOMAIN = {
    "name": "SENTINEL-8004",
    "version": "1",
    "chainId": 8453,           # Base mainnet — override for testnet
    "verifyingContract": "",   # Surge Risk Router address — set from env
}

TRADE_INTENT_TYPES = {
    "TradeIntent": [
        {"name": "agentId",   "type": "uint256"},
        {"name": "pair",      "type": "string"},
        {"name": "direction", "type": "string"},
        {"name": "amount",    "type": "uint256"},
        {"name": "dex",       "type": "string"},
        {"name": "chainId",   "type": "uint256"},
        {"name": "nonce",     "type": "uint256"},
        {"name": "deadline",  "type": "uint256"},
    ]
}


def sign_eip712(intent: TradeIntent, private_key: str) -> str:
    """
    Sign a TradeIntent with EIP-712 and return the hex signature.

    TODO (teammate):
        from eth_account import Account
        from eth_account.messages import encode_typed_data

        structured_data = {
            "types": {"EIP712Domain": [...], "TradeIntent": TRADE_INTENT_TYPES["TradeIntent"]},
            "primaryType": "TradeIntent",
            "domain": DOMAIN,
            "message": intent.model_dump(),
        }
        signable = encode_typed_data(full_message=structured_data)
        signed = Account.sign_message(signable, private_key=private_key)
        return signed.signature.hex()
    """
    raise NotImplementedError("sign_eip712 not yet implemented — teammate to complete")


def verify_eip712(intent: TradeIntent, signature: str, expected_address: str) -> bool:
    """
    Verify an EIP-712 signature (EIP-1271 compatible).

    TODO (teammate): implement using eth_account.recover_message
    """
    raise NotImplementedError("verify_eip712 not yet implemented")
