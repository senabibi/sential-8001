"""
ERC-8004 Validation Registry — post audit/self-correction artifacts.
STUB — to be implemented by teammate.
"""
from __future__ import annotations

from src.models import AuditReport


async def post_validation_artifact(report: AuditReport) -> str:
    """
    Post a Validation Artifact to the ERC-8004 Validation Registry.
    Called by orchestrator after every self-correction event.
    Returns the transaction hash.

    Payload format:
        {
          "trade_id": report.trade_id,
          "outcome": report.outcome,
          "root_cause": report.root_cause,
          "config_updates": [{"param": ..., "old": ..., "new": ...}],
          "timestamp": report.timestamp
        }

    TODO (teammate):
        - Encode payload as bytes
        - Call ValidationRegistry.postArtifact(agentId, keccak256(payload), ipfsUri)
        - Return tx hash
    """
    raise NotImplementedError("post_validation_artifact not yet implemented")
