"""
Risk Manager — purely symbolic, deterministic logic gate.
No LLM involved. Runs before every trade and can veto any signal.

Rules are loaded from config/risk_policy.yaml and re-read on every call
so the Auditor's live config updates take effect immediately.
"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import yaml

from src.models import TradeSignal, RiskDecision, PortfolioState, Direction

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_policy.yaml"


class RiskManager:
    def __init__(self, policy_path: Path = _DEFAULT_POLICY_PATH) -> None:
        self._policy_path = policy_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: TradeSignal,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """
        Run all risk checks. Returns RiskDecision with approved=True only if
        every check passes.
        """
        policy = self._load_policy()
        passed: list[str] = []
        failed: list[str] = []
        adjusted_size: float | None = None

        # 1. HOLD signal — nothing to approve
        if signal.direction == Direction.HOLD:
            return RiskDecision(
                approved=False,
                reason="Signal is HOLD — no trade to execute.",
                checks_passed=["hold_signal"],
                checks_failed=[],
            )

        # 2. Confidence threshold
        min_conf = policy.get("min_signal_confidence", 0.65)
        if signal.confidence < min_conf:
            failed.append(f"confidence_too_low ({signal.confidence:.2f} < {min_conf})")
        else:
            passed.append("confidence_ok")

        # 3. Allowed pair
        allowed = policy.get("allowed_pairs", [])
        if allowed and signal.pair not in allowed:
            failed.append(f"pair_not_allowed ({signal.pair})")
        else:
            passed.append("pair_allowed")

        # 4. Balance floor
        floor = policy.get("balance_floor_usd", 500.0)
        if portfolio.total_balance_usd < floor:
            failed.append(f"balance_below_floor ({portfolio.total_balance_usd:.2f} < {floor})")
        else:
            passed.append("balance_floor_ok")

        # 5. Daily loss limit
        max_daily_loss_pct = policy.get("max_daily_loss_pct", 5.0)
        daily_loss_pct = (
            abs(portfolio.daily_pnl_usd) / portfolio.total_balance_usd * 100
            if portfolio.total_balance_usd > 0 and portfolio.daily_pnl_usd < 0
            else 0.0
        )
        if daily_loss_pct >= max_daily_loss_pct:
            failed.append(
                f"daily_loss_limit_hit ({daily_loss_pct:.2f}% >= {max_daily_loss_pct}%)"
            )
        else:
            passed.append("daily_loss_ok")

        # 6. Max trades per day
        max_trades = policy.get("max_trades_per_day", 20)
        if portfolio.daily_trade_count >= max_trades:
            failed.append(f"max_trades_per_day ({portfolio.daily_trade_count} >= {max_trades})")
        else:
            passed.append("trade_count_ok")

        # 7. Min trade interval
        min_interval = policy.get("min_trade_interval_seconds", 60)
        seconds_since_last = int(time.time()) - portfolio.last_trade_timestamp
        if portfolio.last_trade_timestamp > 0 and seconds_since_last < min_interval:
            failed.append(
                f"trade_too_soon ({seconds_since_last}s < {min_interval}s interval)"
            )
        else:
            passed.append("trade_interval_ok")

        # 8. Position size cap
        max_pos = policy.get("max_position_size_pct", 5.0)
        if signal.size_pct > max_pos:
            # Soft adjustment — cap instead of veto
            adjusted_size = max_pos
            passed.append(f"position_size_capped ({signal.size_pct:.1f}% → {max_pos:.1f}%)")
            logger.warning("Position size capped: %.1f%% → %.1f%%", signal.size_pct, max_pos)
        else:
            min_pos = policy.get("min_position_size_pct", 0.5)
            if signal.size_pct < min_pos:
                failed.append(f"position_too_small ({signal.size_pct:.1f}% < {min_pos}%)")
            else:
                passed.append("position_size_ok")

        # Final decision
        if failed:
            reason = "VETO — " + "; ".join(failed)
            logger.warning("RiskManager VETO: %s", reason)
            return RiskDecision(
                approved=False,
                reason=reason,
                checks_passed=passed,
                checks_failed=failed,
            )

        logger.info("RiskManager APPROVE: %s %s size=%.1f%%", signal.direction.value, signal.pair, adjusted_size or signal.size_pct)
        return RiskDecision(
            approved=True,
            reason="All checks passed.",
            adjusted_size_pct=adjusted_size,
            checks_passed=passed,
            checks_failed=[],
        )

    # ------------------------------------------------------------------
    # Config self-update API (called by Auditor Agent)
    # ------------------------------------------------------------------

    def update_policy(self, param: str, new_value: float) -> tuple[float, float]:
        """
        Update a single numeric parameter in risk_policy.yaml.
        Returns (old_value, new_value) after applying bounds.
        Raises ValueError if param is unknown or value is out of bounds.
        """
        policy = self._load_policy()
        if param not in policy:
            raise ValueError(f"Unknown risk policy param: {param}")

        bounds = policy.get("param_bounds", {}).get(param)
        if bounds:
            lo, hi = bounds
            clamped = max(lo, min(hi, new_value))
            if clamped != new_value:
                logger.warning(
                    "Policy update clamped: %s %.4f → %.4f (bounds [%.2f, %.2f])",
                    param, new_value, clamped, lo, hi,
                )
                new_value = clamped

        old_value = float(policy[param])
        policy[param] = new_value

        with open(self._policy_path, "w") as f:
            yaml.dump(policy, f, default_flow_style=False, sort_keys=False)

        logger.info("Policy updated: %s %.4f → %.4f", param, old_value, new_value)
        return old_value, new_value

    def load_policy(self) -> dict:
        return self._load_policy()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_policy(self) -> dict:
        with open(self._policy_path) as f:
            return yaml.safe_load(f)
