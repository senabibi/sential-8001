"""
Adversarial (Red Team) Agent — stress-tests the Risk Manager.

The Hacker Agent generates malicious / manipulated signals and tries to get them
past the Risk Manager. Every successful bypass reveals a rule gap to patch.

Usage:
    python tests/red_team/adversarial_agent.py --rounds 50
    python tests/red_team/adversarial_agent.py --target risk_manager --rounds 50
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from src.models import Direction, ExecutionLayer, MarketData, OHLCV, PortfolioState, TradeSignal
from src.agents.risk_manager import RiskManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("adversarial_agent")


# ---------------------------------------------------------------------------
# Attack catalogue
# ---------------------------------------------------------------------------

@dataclass
class Attack:
    name: str
    description: str
    signal: TradeSignal
    expected_veto: bool = True   # We EXPECT the risk manager to veto this

    def __str__(self) -> str:
        return f"[{self.name}] {self.description}"


def _fake_portfolio(**kwargs) -> PortfolioState:
    defaults = dict(
        total_balance_usd=10000.0,
        available_balance_usd=9000.0,
        daily_pnl_usd=0.0,
        daily_trade_count=0,
        last_trade_timestamp=0,
    )
    defaults.update(kwargs)
    return PortfolioState(**defaults)


def build_attacks() -> list[Attack]:
    return [
        # 1. Extreme oversize position
        Attack(
            name="oversize_position",
            description="Position size 50% — should be capped/vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=50.0, confidence=0.9, reasoning="Manipulated: huge size"),
        ),
        # 2. Zero confidence
        Attack(
            name="zero_confidence",
            description="Confidence 0.0 — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.0, reasoning="Zero confidence signal"),
        ),
        # 3. Below min confidence
        Attack(
            name="low_confidence",
            description="Confidence 0.3 (below min 0.65) — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.3, reasoning="Low confidence signal"),
        ),
        # 4. Unlisted pair
        Attack(
            name="unlisted_pair",
            description="Trade on unlisted pair DOGE/USD — should be vetoed",
            signal=TradeSignal(pair="DOGE/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.8, reasoning="Unlisted pair injection"),
        ),
        # 5. Balance below floor — portfolio manipulation
        Attack(
            name="balance_floor_bypass",
            description="Portfolio balance $100 (below floor $500) — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.8, reasoning="Legit signal"),
            expected_veto=True,
        ),
        # 6. Daily loss limit hit
        Attack(
            name="daily_loss_limit",
            description="Daily PnL -$800 (8% of $10k, above 5% limit) — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.8, reasoning="After daily loss"),
            expected_veto=True,
        ),
        # 7. Trade too frequent
        Attack(
            name="too_frequent",
            description="Trade placed 5s after last (min interval 60s) — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.8, reasoning="High frequency attempt"),
            expected_veto=True,
        ),
        # 8. Max trades per day hit
        Attack(
            name="max_trades_exceeded",
            description="Daily trade count = 25 (max 20) — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=0.8, reasoning="Over trade count"),
            expected_veto=True,
        ),
        # 9. Size below minimum
        Attack(
            name="below_min_size",
            description="Position size 0.1% (below min 0.5%) — should be vetoed",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=0.1, confidence=0.8, reasoning="Dust trade"),
        ),
        # 10. Inject HOLD as BUY (attempting to bypass HOLD logic)
        Attack(
            name="hold_disguised",
            description="HOLD signal — orchestrator should not trade (not a Risk Manager failure)",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.HOLD, size_pct=1.0, confidence=0.9, reasoning="Hold signal"),
            expected_veto=True,
        ),
        # 11. Negative size (boundary)
        Attack(
            name="negative_size",
            description="Negative size_pct — edge case, must not execute",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=-1.0, confidence=0.8, reasoning="Negative size injection"),
        ),
        # 12. Confidence > 1.0 (boundary)
        Attack(
            name="confidence_overflow",
            description="Confidence 1.5 (out of range) — should still pass if >= min threshold",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=1.0, confidence=1.5, reasoning="Overflow confidence"),
            expected_veto=False,  # Risk manager only checks >= min_conf, so 1.5 may pass — document if it does
        ),
        # 13. Legit signal — sanity check (should NOT be vetoed)
        Attack(
            name="legitimate_signal",
            description="Well-formed legitimate signal — should PASS",
            signal=TradeSignal(pair="BTC/USD", direction=Direction.BUY, size_pct=2.0, confidence=0.8, reasoning="Genuine BTC/USD bullish setup"),
            expected_veto=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Portfolio scenarios per attack
# ---------------------------------------------------------------------------

def portfolio_for_attack(attack_name: str) -> PortfolioState:
    scenarios = {
        "balance_floor_bypass": _fake_portfolio(total_balance_usd=100.0, available_balance_usd=100.0),
        "daily_loss_limit": _fake_portfolio(total_balance_usd=10000.0, daily_pnl_usd=-800.0),
        "too_frequent": _fake_portfolio(last_trade_timestamp=int(time.time()) - 5),
        "max_trades_exceeded": _fake_portfolio(daily_trade_count=25),
    }
    return scenarios.get(attack_name, _fake_portfolio())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    attack: Attack
    vetoed: bool
    reason: str
    passed_as_expected: bool


def run_red_team(rounds: int = 50) -> None:
    risk_manager = RiskManager()
    attacks = build_attacks()

    # Repeat attacks to fill `rounds`
    schedule: list[Attack] = []
    while len(schedule) < rounds:
        schedule.extend(random.choices(attacks, k=min(len(attacks), rounds - len(schedule))))

    results: list[RoundResult] = []
    bypasses: list[RoundResult] = []

    logger.info("Red team starting: %d rounds against RiskManager", rounds)

    for i, attack in enumerate(schedule):
        portfolio = portfolio_for_attack(attack.name)
        decision = risk_manager.evaluate(attack.signal, portfolio)
        vetoed = not decision.approved

        passed_as_expected = vetoed == attack.expected_veto
        result = RoundResult(
            attack=attack,
            vetoed=vetoed,
            reason=decision.reason,
            passed_as_expected=passed_as_expected,
        )
        results.append(result)

        status = "OK" if passed_as_expected else "FAIL"
        logger.info(
            "[%d/%d] %s | vetoed=%s expected_veto=%s | %s",
            i + 1, rounds, attack.name, vetoed, attack.expected_veto, status,
        )

        if not passed_as_expected and attack.expected_veto:
            # Security bypass — should have been vetoed but wasn't
            bypasses.append(result)
            logger.warning("BYPASS DETECTED: %s slipped past Risk Manager!", attack.name)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.passed_as_expected)
    logger.info("=" * 60)
    logger.info("RED TEAM COMPLETE")
    logger.info("Rounds: %d | Passed: %d | Failed: %d", total, passed, total - passed)
    logger.info("Security bypasses (attacks that should have been vetoed but weren't): %d", len(bypasses))

    if bypasses:
        logger.warning("RULE GAPS DETECTED — patch before going live:")
        for r in bypasses:
            logger.warning("  - %s: %s", r.attack.name, r.attack.description)
    else:
        logger.info("No security bypasses found. Risk Manager is solid.")

    # Show unexpected passes (legit signals incorrectly vetoed)
    false_vetos = [r for r in results if not r.passed_as_expected and not r.attack.expected_veto]
    if false_vetos:
        logger.info("False vetos (legit signals incorrectly blocked): %d", len(false_vetos))
        for r in false_vetos:
            logger.info("  - %s: %s", r.attack.name, r.reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL-8004 Red Team Adversarial Testing")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--target", default="risk_manager", help="Target component (default: risk_manager)")
    args = parser.parse_args()
    run_red_team(rounds=args.rounds)
