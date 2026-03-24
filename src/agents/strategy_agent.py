"""
Strategy Agent — generates trade signals using Claude + RAG context + technical indicators.

Inputs:  MarketData (with OHLCV) + pre-computed indicators from retriever
Outputs: TradeSignal (pair, direction, size_pct, confidence, reasoning)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from src.models import MarketData, TradeSignal, Direction
from src.rag.retriever import Retriever
from src.processing.indicators import (
    add_all_indicators,
    classify_market_regime,
)
from src.llm_client import chat_with_fallback, STRATEGY_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SENTINEL-8004's Strategy Agent — a disciplined quantitative trading analyst.

You receive:
  1. Current market snapshot with technical indicators (RSI, EMA, Volume)
  2. RAG context: similar historical market conditions and auditor lessons

Your job is to produce a structured trade signal as valid JSON:

{
  "direction": "buy" | "sell" | "hold",
  "size_pct": <float 0.0-5.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentences>"
}

Decision rules:
- RSI > 70 + price above EMA20 + EMA50 → potential reversal, reduce long bias
- RSI < 30 + price below EMA20 + EMA50 → potential bounce, consider buy
- Volume Z-score > 2 on a down candle → selling pressure, avoid longs
- If RAG shows repeated losses in similar regime → direction: "hold", reduce confidence
- Never suggest size_pct > 3.0 unless confidence > 0.80 AND volume confirms
- If regime is "overbought" or "bearish_breakdown" and you want to BUY → hold instead
- Output ONLY the JSON object, no markdown, no prose before/after
"""


class StrategyAgent:
    def __init__(self, retriever: Retriever, model: str = STRATEGY_MODEL) -> None:
        self._retriever = retriever
        self._model = model

    def generate_signal(self, market_data: MarketData) -> TradeSignal:
        indicators = self._compute_indicators(market_data)
        rag_docs = self._retriever.get_similar_market_conditions(market_data, n=3)
        lessons = self._retriever.get_lessons(
            f"{market_data.pair} {indicators.get('regime', 'unknown')} regime", n=2
        )

        context_str = ""
        if rag_docs:
            context_str += self._retriever.format_context(rag_docs, "Similar historical conditions")
        if lessons:
            context_str += "\n" + self._retriever.format_context(lessons, "Auditor lessons")

        user_message = self._build_user_message(market_data, indicators, context_str)
        raw = self._call_claude(user_message)
        signal = self._parse_signal(raw, market_data.pair)

        logger.info(
            "StrategyAgent → %s %s | conf=%.2f | size=%.1f%% | regime=%s",
            signal.direction.value, signal.pair, signal.confidence,
            signal.size_pct, indicators.get("regime", "?"),
        )
        return signal

    # ------------------------------------------------------------------
    # Indicator computation from MarketData candles
    # ------------------------------------------------------------------

    def _compute_indicators(self, market_data: MarketData) -> dict:
        if len(market_data.candles) < 20:
            return {"regime": "unknown", "rsi14": 50.0, "ema20": market_data.current_price,
                    "ema50": market_data.current_price, "vol_zscore": 0.0, "price_chg_pct": 0.0,
                    "bb_pct": 0.5, "trend_ema20_50": 0.0}

        df = pd.DataFrame([
            {"timestamp": c.timestamp, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in market_data.candles
        ])
        df = add_all_indicators(df)
        last = df.iloc[-1]

        return {
            "rsi14":         round(float(last.get("rsi14", 50)), 2),
            "ema20":         round(float(last.get("ema20", market_data.current_price)), 2),
            "ema50":         round(float(last.get("ema50", market_data.current_price)), 2),
            "ema200":        round(float(last.get("ema200", market_data.current_price)), 2),
            "atr14":         round(float(last.get("atr14", 0)), 2),
            "vol_zscore":    round(float(last.get("vol_zscore", 0)), 2),
            "bb_pct":        round(float(last.get("bb_pct", 0.5)), 3),
            "price_chg_pct": round(float(last.get("price_chg_pct", 0)), 3),
            "trend_ema20_50":round(float(last.get("trend_ema20_50", 0)), 3),
            "regime": classify_market_regime(
                float(last.get("rsi14", 50)),
                float(last.get("price_chg_pct", 0)),
                float(last.get("vol_zscore", 0)),
                float(last.get("bb_pct", 0.5)),
                float(last.get("trend_ema20_50", 0)),
            ),
        }

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_user_message(self, md: MarketData, ind: dict, context: str) -> str:
        last5 = [f"{c.close:.2f}" for c in md.candles[-5:]]
        trend_desc = self._trend_description(md)

        lines = [
            f"Pair:         {md.pair}",
            f"Current:      {md.current_price:.4f} USD",
            f"Last 5 closes: {', '.join(last5)}",
            f"Trend (20H):  {trend_desc}",
            "",
            "=== Technical Indicators ===",
            f"RSI(14):      {ind['rsi14']}  {'⚠ overbought' if ind['rsi14']>70 else '⚠ oversold' if ind['rsi14']<30 else 'neutral'}",
            f"EMA(20):      {ind['ema20']}  |  EMA(50): {ind['ema50']}",
            f"EMA gap 20/50:{ind['trend_ema20_50']:+.2f}%  ({'bullish' if ind['trend_ema20_50']>0 else 'bearish'} crossover)",
            f"Volume Z:     {ind['vol_zscore']:+.2f}  ({'high volume' if abs(ind['vol_zscore'])>2 else 'normal'})",
            f"Bollinger %B: {ind['bb_pct']:.2f}",
            f"ATR(14):      {ind['atr14']:.2f}",
            f"Regime:       {ind['regime']}",
            f"1H chg:       {ind['price_chg_pct']:+.2f}%",
        ]

        if context:
            lines += ["", context]

        lines += ["", "Provide your signal as JSON:"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call + parsing
    # ------------------------------------------------------------------

    def _call_claude(self, user_message: str) -> str:
        return chat_with_fallback(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            model=self._model,
            max_tokens=512,
            temperature=0.2,
        )

    def _parse_signal(self, raw: str, pair: str) -> TradeSignal:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            data = json.loads(cleaned)
            return TradeSignal(
                pair=pair,
                direction=Direction(data["direction"]),
                size_pct=min(float(data["size_pct"]), 5.0),
                confidence=max(0.0, min(1.0, float(data["confidence"]))),
                reasoning=str(data["reasoning"]),
                rag_context_used=True,
                strategy_model=self._model,
            )
        except Exception as e:
            logger.error("Signal parse error: %s | raw: %.200s", e, raw)
            return TradeSignal(
                pair=pair,
                direction=Direction.HOLD,
                size_pct=0.0,
                confidence=0.0,
                reasoning=f"Parse error — HOLD. Raw: {raw[:80]}",
                strategy_model=self._model,

            )

    @staticmethod
    def _trend_description(md: MarketData) -> str:
        if len(md.candles) < 2:
            return "unknown"
        closes = [c.close for c in md.candles[-20:]]
        pct = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] != 0 else 0
        arrow = "↑" if pct > 0 else "↓"
        return f"{arrow} {pct:+.2f}%"
