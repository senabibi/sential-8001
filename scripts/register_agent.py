"""
Register SENTINEL-8004 on-chain as an ERC-8004 Agent Identity.
STUB — requires teammate's onchain/ implementation.

Usage:
    python scripts/register_agent.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_agent")

REGISTRATION_PATH = Path(__file__).resolve().parents[1] / "config" / "agent_registration.json"


async def main() -> None:
    reg = json.loads(REGISTRATION_PATH.read_text())
    logger.info("Registering agent: %s v%s", reg["name"], reg["version"])

    try:
        from src.onchain.identity import mint_agent_identity
        private_key = os.environ["AGENT_WALLET_PRIVATE_KEY"]
        token_id = await mint_agent_identity(
            name=reg["name"],
            capabilities=reg["capabilities"],
            metadata_uri="",   # TODO: upload to IPFS first
            private_key=private_key,
        )
        reg["erc8004"]["identity_token_id"] = token_id
        REGISTRATION_PATH.write_text(json.dumps(reg, indent=2))
        logger.info("Agent registered. Token ID: %d", token_id)
    except NotImplementedError:
        logger.warning("onchain/identity.py not yet implemented — skipping on-chain registration")


if __name__ == "__main__":
    asyncio.run(main())
