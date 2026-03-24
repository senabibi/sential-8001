"""
Claim sandbox capital from the Hackathon Capital Vault.
STUB — requires teammate's onchain/ implementation.

Usage:
    python scripts/claim_sandbox_capital.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("claim_sandbox_capital")


async def main() -> None:
    agent_id = os.getenv("CAPITAL_VAULT_AGENT_ID")
    if not agent_id:
        logger.error("CAPITAL_VAULT_AGENT_ID not set in .env")
        return

    logger.info("Claiming sandbox capital for agent_id=%s", agent_id)
    logger.warning("onchain Capital Vault claim not yet implemented — teammate to complete")
    # TODO (teammate): call CapitalVault.claimSandboxCapital(agent_id)


if __name__ == "__main__":
    asyncio.run(main())
