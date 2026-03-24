"""
Digital Twin Simulator — paper trading with real historical data.

Replays historical BTC/USD and ETH/USD windows through the full agent loop:
  Observe → Hypothesize → Gate (RiskManager) → Execute → Audit → Self-Correct

Every trade triggers the Auditor Agent which may update config/risk_policy.yaml.
After N windows the resulting risk_policy.yaml represents the agent's learned state.

Scenarios:
  flash_crash   — 5 worst single-candle drops from history
  bull_pump     — 5 best single-candle spikes from history
  sideways      — 10 low-volatility windows
  custom_dates  — specify start/end date manually
  random_N      — N randomly sampled 100-candle windows

Usage:
    python scripts/simulator.py --scenario flash_crash
    python scripts/simulator.py --scenario random_N --n 100 --capital 10000
    python scripts/simulator.py --scenario custom_dates --since 2024-08-01 --until 2024-08-31
    python scripts/simulator.py --pair ETH/USD --scenario flash_crash
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import pandas as pd

from src.models import (
    ClosedTrade, Direction, ExecutionLayer, MarketData, OHLCV, Order,
    ExecutionResult, PortfolioState, TradeMode,
)
from src.core.base_worker import BaseWorker
from src.agents.strategy_agent import StrategyAgent
from src.agents.risk_manager import RiskManager
from src.agents.auditor_agent import AuditorAgent
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.processing.cleaner import (
    load_ohlcv_df, detect_significant_events,
    get_context_window, pair_from_filename,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("simulator")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"

PAIR_TO_FILE = {
    "BTC/USD": "XBTUSD_60.csv",
    "ETH/USD": "ETHUSD_60.csv",
}

# Window size fed to the strategy agent per simulation step
WINDOW_SIZE = 100  # candles (100H ≈ 4 days of hourly data)


# ---------------------------------------------------------------------------
# Simulation execution worker
# ---------------------------------------------------------------------------

class SimulationWorker(BaseWorker):
    """
    Paper-trading worker that replays injected MarketData.
    Simulates trade outcomes using stop-loss / take-profit logic
    against actual subsequent candles from the historical data.
    """

    def __init__(self, initial_capital: float = 10000.0) -> None:
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_trade_ts = 0
        self._market_data: Optional[MarketData] = None
        self._outcome_candles: list[OHLCV] = []   # candles after entry — for realistic PnL
        self._all_trades: list[dict] = []

    @property
    def name(self) -> str:
        return "kraken"

    def set_window(self, market_data: MarketData, outcome_candles: list[OHLCV]) -> None:
        """Inject the current window + subsequent candles for outcome simulation."""
        self._market_data = market_data
        self._outcome_candles = outcome_candles
        # Reset daily state for each new "day"
        self._daily_pnl = 0.0
        self._daily_trades = 0

    async def get_market_data(self, pair: str, num_candles: int = 100) -> MarketData:
        if self._market_data is None:
            raise ValueError("No market data injected — call set_window() first")
        return self._market_data

    async def get_portfolio(self) -> PortfolioState:
        return PortfolioState(
            total_balance_usd=self._capital,
            available_balance_usd=self._capital,
            daily_pnl_usd=self._daily_pnl,
            daily_trade_count=self._daily_trades,
            last_trade_timestamp=self._last_trade_ts,
        )

    async def execute_order(self, order: Order) -> ExecutionResult:
        if self._market_data is None:
            raise ValueError("No market data")

        entry_price = self._market_data.current_price
        alloc_usd = self._capital * (order.size_pct / 100)
        size = alloc_usd / entry_price if entry_price > 0 else 0.0

        pnl_usd, exit_price = self._simulate_outcome(
            entry_price=entry_price,
            direction=order.direction,
            stop_loss_pct=order.stop_loss_pct or 2.0,
            take_profit_pct=order.take_profit_pct or 4.0,
            size_usd=alloc_usd,
        )

        self._capital += pnl_usd
        self._daily_pnl += pnl_usd
        self._daily_trades += 1
        self._last_trade_ts = int(time.time())

        trade_id = order.trade_id or str(uuid.uuid4())[:8]
        self._all_trades.append({
            "trade_id": trade_id,
            "pair": order.pair,
            "direction": order.direction.value,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": alloc_usd,
            "pnl_usd": pnl_usd,
            "pnl_pct": (pnl_usd / alloc_usd * 100) if alloc_usd > 0 else 0,
        })

        return ExecutionResult(
            success=True,
            trade_id=trade_id,
            pair=order.pair,
            direction=order.direction,
            executed_price=entry_price,
            executed_size=size,
            execution_layer=ExecutionLayer.KRAKEN,
        )

    async def close_position(self, pair: str, sandbox: bool = True) -> ExecutionResult:
        raise NotImplementedError

    def _simulate_outcome(
        self,
        entry_price: float,
        direction: Direction,
        stop_loss_pct: float,
        take_profit_pct: float,
        size_usd: float,
    ) -> tuple[float, float]:
        """
        Walk through outcome_candles candle by candle.
        Return (pnl_usd, exit_price) at whichever event fires first:
          stop-loss, take-profit, or end of window.
        """
        if not self._outcome_candles:
            return 0.0, entry_price

        sl_price = entry_price * (1 - stop_loss_pct / 100) if direction == Direction.BUY else entry_price * (1 + stop_loss_pct / 100)
        tp_price = entry_price * (1 + take_profit_pct / 100) if direction == Direction.BUY else entry_price * (1 - take_profit_pct / 100)

        for candle in self._outcome_candles:
            if direction == Direction.BUY:
                if candle.low <= sl_price:
                    exit_p = sl_price
                    pnl = (exit_p - entry_price) / entry_price * size_usd
                    return pnl, exit_p
                if candle.high >= tp_price:
                    exit_p = tp_price
                    pnl = (exit_p - entry_price) / entry_price * size_usd
                    return pnl, exit_p
            else:  # SELL
                if candle.high >= sl_price:
                    exit_p = sl_price
                    pnl = (entry_price - exit_p) / entry_price * size_usd
                    return pnl, exit_p
                if candle.low <= tp_price:
                    exit_p = tp_price
                    pnl = (entry_price - exit_p) / entry_price * size_usd
                    return pnl, exit_p

        # No SL/TP hit — exit at last candle close
        exit_p = self._outcome_candles[-1].close
        if direction == Direction.BUY:
            pnl = (exit_p - entry_price) / entry_price * size_usd
        else:
            pnl = (entry_price - exit_p) / entry_price * size_usd
        return pnl, exit_p

    def summary(self) -> dict:
        wins = [t for t in self._all_trades if t["pnl_usd"] > 0]
        losses = [t for t in self._all_trades if t["pnl_usd"] < 0]
        total_pnl = sum(t["pnl_usd"] for t in self._all_trades)
        avg_win = sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0
        return {
            "total_trades":     len(self._all_trades),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         len(wins) / len(self._all_trades) if self._all_trades else 0,
            "total_pnl_usd":    total_pnl,
            "final_capital":    self._capital,
            "initial_capital":  self._initial_capital,
            "pnl_pct":          (self._capital - self._initial_capital) / self._initial_capital * 100,
            "avg_win_usd":      avg_win,
            "avg_loss_usd":     avg_loss,
            "profit_factor":    abs(sum(t["pnl_usd"] for t in wins) / sum(t["pnl_usd"] for t in losses)) if losses else float("inf"),
        }


# ---------------------------------------------------------------------------
# Window builders
# ---------------------------------------------------------------------------

def _df_to_candles(df: pd.DataFrame) -> list[OHLCV]:
    return [
        OHLCV(timestamp=int(r.timestamp), open=float(r.open), high=float(r.high),
              low=float(r.low), close=float(r.close), volume=float(r.volume))
        for r in df.itertuples()
    ]


def _make_market_data(pair: str, candles: list[OHLCV]) -> MarketData:
    return MarketData(
        pair=pair,
        candles=candles,
        current_price=candles[-1].close,
        volume_24h=sum(c.volume for c in candles[-24:]),
        source=ExecutionLayer.KRAKEN,
    )


def build_windows(
    df: pd.DataFrame,
    pair: str,
    scenario: str,
    n_windows: int = 100,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    outcome_window: int = 48,
) -> list[tuple[MarketData, list[OHLCV], str]]:
    """
    Returns list of (market_data_window, outcome_candles, label) tuples.
    market_data_window: WINDOW_SIZE candles up to the anchor point
    outcome_candles:    next outcome_window candles (for PnL simulation)
    label:              human description of this window
    """
    if since:
        since_ts = pd.Timestamp(since, tz="UTC") if since.tzinfo is None else pd.Timestamp(since).tz_convert("UTC")
        df = df[df["datetime"] >= since_ts]
    if until:
        until_ts = pd.Timestamp(until, tz="UTC") if until.tzinfo is None else pd.Timestamp(until).tz_convert("UTC")
        df = df[df["datetime"] <= until_ts]
    df = df.reset_index(drop=True)

    windows: list[tuple[MarketData, list[OHLCV], str]] = []
    all_candles = _df_to_candles(df)

    def _slice(anchor: int) -> tuple[MarketData, list[OHLCV], str]:
        start = max(0, anchor - WINDOW_SIZE + 1)
        w_candles = all_candles[start : anchor + 1]
        o_candles = all_candles[anchor + 1 : anchor + 1 + outcome_window]
        dt = df["datetime"].iloc[anchor].strftime("%Y-%m-%d %H:%M") if anchor < len(df) else "?"
        md = _make_market_data(pair, w_candles)
        return md, o_candles, dt

    if scenario == "flash_crash":
        events = detect_significant_events(df, price_drop_threshold=-5.0)
        drops = events[events["event_type"].str.contains("price_drop", na=False)]
        drops = drops.nsmallest(min(10, len(drops)), "price_chg_pct")
        for _, row in drops.iterrows():
            anchor = int(row["df_index"])
            if anchor >= WINDOW_SIZE:
                windows.append(_slice(anchor))

    elif scenario == "bull_pump":
        events = detect_significant_events(df, price_spike_threshold=5.0)
        pumps = events[events["event_type"].str.contains("price_spike", na=False)]
        pumps = pumps.nlargest(min(10, len(pumps)), "price_chg_pct")
        for _, row in pumps.iterrows():
            anchor = int(row["df_index"])
            if anchor >= WINDOW_SIZE:
                windows.append(_slice(anchor))

    elif scenario == "sideways":
        # Low absolute price change windows
        from src.processing.indicators import compute_rsi
        step = max(1, len(df) // 20)
        for anchor in range(WINDOW_SIZE, len(df) - outcome_window, step):
            w = df.iloc[anchor - WINDOW_SIZE : anchor + 1]
            chg = abs((w["close"].iloc[-1] - w["close"].iloc[0]) / w["close"].iloc[0] * 100)
            if chg < 3.0:
                windows.append(_slice(anchor))
                if len(windows) >= 10:
                    break

    elif scenario == "custom_dates":
        # Use entire filtered range, stepping by WINDOW_SIZE
        for anchor in range(WINDOW_SIZE, len(df) - outcome_window, WINDOW_SIZE // 2):
            windows.append(_slice(anchor))

    else:  # random_N
        valid = list(range(WINDOW_SIZE, len(df) - outcome_window))
        if not valid:
            logger.warning("Not enough data for random windows")
            return []
        chosen = random.sample(valid, min(n_windows, len(valid)))
        for anchor in sorted(chosen):
            windows.append(_slice(anchor))

    logger.info("Built %d windows for scenario '%s' on %s", len(windows), scenario, pair)
    return windows


# ---------------------------------------------------------------------------
# Single simulation cycle
# ---------------------------------------------------------------------------

async def run_window(
    window_idx: int,
    market_data: MarketData,
    outcome_candles: list[OHLCV],
    window_label: str,
    worker: SimulationWorker,
    strategy_agent: StrategyAgent,
    risk_manager: RiskManager,
    auditor_agent: AuditorAgent,
    pair: str,
) -> dict:
    worker.set_window(market_data, outcome_candles)
    portfolio = await worker.get_portfolio()

    signal = strategy_agent.generate_signal(market_data)
    decision = risk_manager.evaluate(signal, portfolio)

    result = {"window": window_idx, "label": window_label, "direction": signal.direction.value,
               "confidence": signal.confidence, "pnl_usd": 0.0, "outcome": "veto",
               "config_updates": 0}

    if not decision.approved:
        logger.info("[%d] %s VETO: %s", window_idx, window_label, decision.reason)
        return result

    size = decision.adjusted_size_pct or signal.size_pct
    policy = risk_manager.load_policy()
    order = Order(
        pair=pair,
        direction=signal.direction,
        size_pct=size,
        execution_layer=ExecutionLayer.KRAKEN,
        sandbox=True,
        trade_id=str(uuid.uuid4())[:8],
        stop_loss_pct=policy.get("stop_loss_pct"),
        take_profit_pct=policy.get("take_profit_pct"),
    )

    exec_result = await worker.execute_order(order)

    # Determine actual PnL from the trade record
    trade_record = worker._all_trades[-1]
    pnl_usd = trade_record["pnl_usd"]
    exit_price = trade_record["exit_price"]

    entry_price = exec_result.executed_price
    pnl_pct = (pnl_usd / (exec_result.executed_size * entry_price) * 100) if exec_result.executed_size > 0 else 0

    closed = ClosedTrade(
        trade_id=exec_result.trade_id,
        pair=pair,
        direction=signal.direction,
        entry_price=entry_price,
        exit_price=exit_price,
        size=exec_result.executed_size,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        execution_layer=ExecutionLayer.KRAKEN,
        entry_timestamp=exec_result.timestamp,
        exit_timestamp=int(time.time()),
        signal=signal,
        order=order,
        result=exec_result,
    )

    audit = auditor_agent.audit(closed)

    outcome_label = "profit" if pnl_usd > 0 else "loss"
    logger.info(
        "[%d] %s | %s %s @ %.2f | PnL %+.2f USD | %s | %d policy updates",
        window_idx, window_label, signal.direction.value, pair, entry_price,
        pnl_usd, outcome_label, len(audit.config_updates),
    )

    for cu in audit.config_updates:
        logger.info("  Policy: %s %.4f → %.4f (%s)", cu.param, cu.old_value, cu.new_value, cu.reason)

    result.update({
        "pnl_usd": pnl_usd,
        "outcome": outcome_label,
        "config_updates": len(audit.config_updates),
    })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_simulation(
    scenario: str,
    pair: str = "BTC/USD",
    capital: float = 10000.0,
    n_windows: int = 100,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> None:
    filename = PAIR_TO_FILE.get(pair)
    if not filename:
        logger.error("Unknown pair: %s. Supported: %s", pair, list(PAIR_TO_FILE.keys()))
        return

    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        logger.error("Data file not found: %s — run prepare_data.py first", csv_path)
        return

    logger.info("Loading %s from %s...", pair, csv_path.name)
    df = load_ohlcv_df(csv_path, since=since, until=until)
    if df.empty:
        logger.error("No data after filtering")
        return

    windows = build_windows(df, pair, scenario, n_windows=n_windows, since=since, until=until)
    if not windows:
        logger.error("No windows built for scenario '%s'", scenario)
        return

    # Build agent stack
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    risk_manager = RiskManager()
    strategy_agent = StrategyAgent(retriever)
    auditor_agent = AuditorAgent(risk_manager, retriever)
    worker = SimulationWorker(initial_capital=capital)

    logger.info("Simulation: %d windows | %s | %s | capital=$%.0f",
                len(windows), scenario, pair, capital)
    logger.info("-" * 60)

    all_results = []
    for i, (md, outcome, label) in enumerate(windows):
        res = await run_window(
            window_idx=i + 1,
            market_data=md,
            outcome_candles=outcome,
            window_label=label,
            worker=worker,
            strategy_agent=strategy_agent,
            risk_manager=risk_manager,
            auditor_agent=auditor_agent,
            pair=pair,
        )
        all_results.append(res)

    # Final summary
    summary = worker.summary()
    traded = [r for r in all_results if r["outcome"] != "veto"]
    vetos = [r for r in all_results if r["outcome"] == "veto"]
    total_policy_updates = sum(r["config_updates"] for r in all_results)

    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE — %s | %s", scenario, pair)
    logger.info("Windows:       %d", len(windows))
    logger.info("Trades placed: %d", len(traded))
    logger.info("Vetoed:        %d", len(vetos))
    logger.info("Wins / Losses: %d / %d  (%.1f%% win rate)",
                summary["wins"], summary["losses"], summary["win_rate"] * 100)
    logger.info("Total PnL:     $%+.2f  (%+.1f%%)", summary["total_pnl_usd"], summary["pnl_pct"])
    logger.info("Final capital: $%.2f  (started $%.2f)", summary["final_capital"], summary["initial_capital"])
    logger.info("Avg win / loss: $%+.2f / $%.2f", summary["avg_win_usd"], summary["avg_loss_usd"])
    logger.info("Profit factor: %.2f", summary["profit_factor"])
    logger.info("Policy updates (self-corrections): %d", total_policy_updates)
    logger.info("RAG lessons stored: %d", vector_store.collection_count("lessons"))
    logger.info("")
    logger.info("Learned config is in config/risk_policy.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL-8004 Simulator")
    parser.add_argument("--scenario", required=True,
                        choices=["flash_crash", "bull_pump", "sideways", "custom_dates", "random_N"])
    parser.add_argument("--pair", default="BTC/USD", choices=["BTC/USD", "ETH/USD"])
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--n", type=int, default=100, help="Windows for random_N")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD start date")
    parser.add_argument("--until", default=None, help="YYYY-MM-DD end date")
    args = parser.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.since else None
    until = datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.until else None

    asyncio.run(run_simulation(args.scenario, args.pair, args.capital, args.n, since, until))
