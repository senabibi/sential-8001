"""
Auditor Agent — post-trade root-cause analysis and self-correction.

Uses Groq (llama-3.3-70b) to analyse a closed trade, extract a lesson,
optionally update risk_policy.yaml, and store the lesson in RAG.

Every config change is returned as a ConfigUpdate for the orchestrator
to post as a Validation Artifact on-chain.
"""
from __future__ import annotations

import json
import logging

from src.models import ClosedTrade, AuditReport, ConfigUpdate
from src.agents.risk_manager import RiskManager
from src.rag.retriever import Retriever
from src.llm_client import chat_with_fallback, AUDITOR_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SENTINEL-8004's Auditor Agent — a rigorous post-trade analyst.

After each trade closes you receive the trade record, its context, and the current risk policy.
Your job:
1. Identify the root cause of any loss (or confirm what made a profit sustainable)
2. Extract a concise lesson (1-2 sentences) for future trades
3. Optionally recommend ONE risk_policy.yaml parameter update

Respond with valid JSON:
{
  "root_cause": "<1 sentence>",
  "lesson": "<1-2 sentences for RAG storage>",
  "config_update": {                      // null if no update needed
    "param": "<param_name>",
    "new_value": <float>,
    "reason": "<why>"
  }
}

Only recommend a config update if the trade clearly reveals a systematic miscalibration.
Output ONLY the JSON object.
"""


class AuditorAgent:
    def __init__(
        self,
        risk_manager: RiskManager,
        retriever: Retriever,
        model: str = AUDITOR_MODEL,
    ) -> None:
        self._risk_manager = risk_manager
        self._retriever = retriever
        self._model = model

    def audit(self, trade: ClosedTrade) -> AuditReport:
        """
        Full audit cycle:
        1. Build prompt with trade + RAG context + current policy
        2. Call GPT-4o
        3. Parse response
        4. Apply config update (if any, within bounds)
        5. Store lesson in RAG
        6. Store trade in RAG
        """
        policy = self._risk_manager.load_policy()
        similar_lessons = self._retriever.get_lessons(
            f"{trade.pair} {trade.direction.value} trade", n=2
        )
        context = self._retriever.format_context(similar_lessons, "Similar past lessons")

        user_message = self._build_prompt(trade, policy, context)
        raw = self._call_gpt4o(user_message)
        parsed = self._parse_response(raw)

        config_updates: list[ConfigUpdate] = []
        if parsed.get("config_update"):
            cu = parsed["config_update"]
            try:
                param = cu["param"]
                new_val = float(cu["new_value"])
                old_val, applied_val = self._risk_manager.update_policy(param, new_val)
                config_updates.append(
                    ConfigUpdate(
                        param=param,
                        old_value=old_val,
                        new_value=applied_val,
                        reason=cu.get("reason", ""),
                    )
                )
                logger.info("Auditor updated policy: %s %.4f → %.4f", param, old_val, applied_val)
            except Exception as e:
                logger.error("Config update failed: %s", e)

        lesson = parsed.get("lesson", "")
        if lesson:
            self._retriever.store_lesson(
                lesson,
                metadata={
                    "trade_id": trade.trade_id,
                    "pair": trade.pair,
                    "pnl_usd": trade.pnl_usd,
                    "direction": trade.direction.value,
                },
            )

        self._retriever.store_trade(trade)

        outcome = "profit" if trade.pnl_usd > 0 else "loss" if trade.pnl_usd < 0 else "break_even"
        report = AuditReport(
            trade_id=trade.trade_id,
            outcome=outcome,
            pnl_usd=trade.pnl_usd,
            root_cause=parsed.get("root_cause", "Unknown"),
            lesson=lesson,
            config_updates=config_updates,
            rag_lesson_stored=bool(lesson),
            auditor_model=self._model,
        )

        logger.info(
            "Audit complete: %s | %s | %s | %d policy updates",
            trade.trade_id, outcome, report.root_cause, len(config_updates),
        )
        return report

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_prompt(self, trade: ClosedTrade, policy: dict, context: str) -> str:
        lines = [
            f"Trade ID: {trade.trade_id}",
            f"Pair: {trade.pair}",
            f"Direction: {trade.direction.value}",
            f"Entry: {trade.entry_price:.4f}  Exit: {trade.exit_price:.4f}",
            f"PnL: {trade.pnl_usd:+.2f} USD ({trade.pnl_pct:+.2f}%)",
            f"Execution layer: {trade.execution_layer.value}",
            "",
            "Current risk policy (relevant params):",
            f"  stop_loss_pct: {policy.get('stop_loss_pct')}",
            f"  take_profit_pct: {policy.get('take_profit_pct')}",
            f"  max_position_size_pct: {policy.get('max_position_size_pct')}",
            f"  min_signal_confidence: {policy.get('min_signal_confidence')}",
        ]

        if trade.signal:
            lines += [
                "",
                "Signal that triggered this trade:",
                f"  confidence: {trade.signal.confidence:.2f}",
                f"  reasoning: {trade.signal.reasoning}",
            ]

        if context:
            lines += ["", context]

        lines += ["", "Provide your audit as JSON:"]
        return "\n".join(lines)

    def _call_gpt4o(self, user_message: str) -> str:
        return chat_with_fallback(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            model=self._model,
            max_tokens=512,
            temperature=0.3,
        )

    @staticmethod
    def _parse_response(raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            return json.loads(cleaned)
        except Exception as e:
            logger.error("AuditorAgent parse error: %s | raw: %s", e, raw[:200])
            return {
                "root_cause": "Parse error",
                "lesson": "",
                "config_update": None,
            }
