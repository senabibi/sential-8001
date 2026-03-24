# SENTINEL-8004
### Autonomous Self-Correcting Trading Agent

> A multi-agent, self-optimizing trading system competing across **both hackathon tracks simultaneously**: Kraken CLI (CeFi execution) + ERC-8004 / DeFi Track (Surge on-chain trust layer) + Aerodrome Finance (on-chain liquidity execution on Base).

**Hackathon:** lablab.ai Autonomous Trading Agents Hackathon
**Tracks:** Kraken CLI Track + ERC-8004 / DeFi Track (Surge) — both active
**LLM Provider:** Groq (llama-3.3-70b-versatile + llama-3.1-8b-instant)
**Embeddings:** Local sentence-transformers (all-MiniLM-L6-v2, no API key needed)

---

## Table of Contents

- [Overview](#overview)
- [What's Built & Working](#whats-built--working)
- [Architecture: The Triple-Helix Logic](#architecture-the-triple-helix-logic)
- [Directory Structure](#directory-structure)
- [The Self-Correction Loop](#the-self-correction-loop)
- [Technical Indicators](#technical-indicators)
- [RAG Memory: Synthetic Experience](#rag-memory-synthetic-experience)
- [Simulation Results](#simulation-results)
- [Execution Layers: Kraken + Aerodrome](#execution-layers-kraken--aerodrome)
- [ERC-8004 Trust & Identity Layer](#erc-8004-trust--identity-layer)
- [EIP-712 Trade Intent Signing](#eip-712-trade-intent-signing)
- [Risk Router Integration (Surge)](#risk-router-integration-surge)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Division of Work](#division-of-work)
- [Hackathon Track Compliance](#hackathon-track-compliance)
- [Roadmap](#roadmap)

---

## Overview

SENTINEL-8004 is a multi-agent, dual-track trading system built around three core principles:

1. **Memory** — Remembers past market crises and its own trade history via a RAG layer seeded with real Kraken historical data (2022–2025). Before its first live trade, the agent has already "experienced" the LUNA collapse, FTX crash, 2024 halving correction, and the August 2024 yen carry trade flash crash.

2. **Safety** — Hard, deterministic risk rules run independently of any LLM, enforced both in Python and on-chain via the Surge Risk Router. The Risk Manager checks 8 conditions on every trade: confidence threshold, pair whitelist, balance floor, daily loss limit, trade frequency, position size, leverage, and drawdown.

3. **Self-Improvement** — After every trade, the Auditor Agent (Groq/llama) performs root-cause analysis and autonomously updates `config/risk_policy.yaml` within pre-defined safety bounds. Every optimization is signed on-chain.

The agent operates across two execution environments in parallel:
- **CeFi (Kraken CLI):** Spot and derivatives trading via Kraken's AI-native Rust binary
- **DeFi (Aerodrome Finance on Base):** On-chain swaps and LP positions via Aerodrome's AMM, submitted through the Surge Risk Router

All trust signals, agent identity, reputation, and validation artifacts are anchored on-chain via **ERC-8004** on Base L2.

---

## What's Built & Working

### Agent Layer (complete — this repo)

| Component | Status | Description |
|-----------|--------|-------------|
| `src/models.py` | ✅ Done | All shared Pydantic models: TradeSignal, Order, ClosedTrade, AuditReport, TradeIntent, etc. |
| `src/processing/indicators.py` | ✅ Done | RSI(14), EMA(20/50/200), ATR(14), Volume Z-score, Bollinger %B, regime classifier |
| `src/processing/cleaner.py` | ✅ Done | Kraken CSV loader (auto-detects header), event detection, context windowing |
| `src/processing/embedder.py` | ✅ Done | Chunks OHLCV into RAG-ready narrative paragraphs |
| `src/rag/vector_store.py` | ✅ Done | ChromaDB wrapper with sqlite3 patch for Python 3.9 |
| `src/rag/retriever.py` | ✅ Done | Local sentence-transformers embeddings, semantic search, store/retrieve lessons |
| `src/agents/strategy_agent.py` | ✅ Done | Groq (llama-3.3-70b) + RAG context + live technical indicators → TradeSignal |
| `src/agents/risk_manager.py` | ✅ Done | 8-rule deterministic gate, live-reloads risk_policy.yaml, supports auditor updates |
| `src/agents/auditor_agent.py` | ✅ Done | Groq post-trade root-cause analysis, updates risk_policy.yaml, stores lessons in RAG |
| `src/core/orchestrator.py` | ✅ Done | Main Observe→Gate→Execute→Audit loop |
| `src/core/base_worker.py` | ✅ Done | Abstract interface for execution workers (teammate implements) |
| `src/llm_client.py` | ✅ Done | Groq client with multi-key rotation on rate limits |
| `scripts/prepare_data.py` | ✅ Done | Copies & validates Kraken OHLCV files, confirms key historical events present |
| `scripts/bootstrap_memory.py` | ✅ Done | Event-based RAG seeding with GPT-generated lessons from real crisis data |
| `scripts/simulator.py` | ✅ Done | Paper trading with real historical windows, SL/TP simulation, self-correction |
| `tests/red_team/adversarial_agent.py` | ✅ Done | 13 attack vectors vs Risk Manager |

### RAG Memory (live state after bootstrap)

| Collection | Documents | Content |
|-----------|-----------|---------|
| `market_cycles` | **2,918** | BTC/USD + ETH/USD hourly regime chunks (2022→2025) |
| `lessons` | **380** | LLM-generated lessons from real crisis events |
| `trade_history` | **1** | Aug 2024 yen carry trade flash crash (self-correction applied) |

### Teammate's Layer (stubs ready for integration)

| Component | Who | Description |
|-----------|-----|-------------|
| `src/core/kraken_worker.py` | Teammate | Kraken CLI subprocess wrapper |
| `src/core/aerodrome_worker.py` | Teammate | Aerodrome web3.py + Risk Router |
| `src/onchain/signing.py` | Teammate | EIP-712 typed data signing |
| `src/onchain/identity.py` | Teammate | ERC-8004 NFT mint |
| `src/onchain/reputation.py` | Teammate | Reputation Registry posts |
| `src/onchain/validator.py` | Teammate | Validation Artifact posts |
| `src/onchain/risk_router.py` | Teammate | Surge Risk Router submission |
| `src/onchain/wallet.py` | Teammate | EIP-1271 smart contract wallet |

---

## Architecture: The Triple-Helix Logic

```
                    ┌──────────────────────────────────┐
                    │           ORCHESTRATOR            │
                    │         (orchestrator.py)         │
                    └────┬─────────────┬────────────────┘
                         │             │
          ┌──────────────▼──┐   ┌──────▼─────────────────┐
          │  MEMORY LAYER   │   │     SAFETY LAYER        │
          │  (RAG Engine)   │   │  (Symbolic + On-Chain)  │
          │                 │   │                         │
          │ 2,918 market    │   │ - 8 deterministic rules │
          │ cycle chunks    │   │ - Surge Risk Router     │
          │ 380 lessons     │   │ - EIP-712 intent sign   │
          │ (real crises)   │   └──────────┬──────────────┘
          └────────┬────────┘              │
                   └──────────┬────────────┘
                              │
                 ┌────────────▼──────────────┐
                 │      STRATEGY AGENT       │
                 │ Groq llama-3.3-70b +      │
                 │ RSI / EMA / Volume / RAG  │
                 └────────────┬──────────────┘
                              │
                 ┌────────────▼──────────────┐
                 │       RISK MANAGER        │
                 │     APPROVE or VETO       │
                 └────────┬──────────┬───────┘
                          │          │
          ┌───────────────▼──┐  ┌────▼────────────────────┐
          │  CeFi EXECUTION  │  │   DeFi EXECUTION         │
          │  Kraken CLI      │  │   Aerodrome Finance      │
          │  (MCP Server)    │  │   (Base L2, Risk Router) │
          └────────┬─────────┘  └─────────────┬────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │        AUDITOR AGENT         │
                 │  Groq llama-3.3-70b          │
                 │  PnL root-cause → YAML update│
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │        ERC-8004 LAYER        │
                 │  Identity · Reputation       │
                 │  Validation · On-Chain Log   │
                 └──────────────────────────────┘
```

---

## Directory Structure

```
sentinel-8004/
│
├── config/
│   ├── risk_policy.yaml           # Hard limits — auto-updated by Auditor
│   ├── model_routing.yaml         # LLM routing: Groq models + local embedder
│   └── agent_registration.json    # ERC-8004 Agent Identity metadata
│
├── src/
│   ├── models.py                  # All shared Pydantic data models
│   ├── llm_client.py              # Groq client factory with key rotation
│   │
│   ├── core/
│   │   ├── orchestrator.py        # Main loop: Observe → Gate → Execute → Audit
│   │   ├── base_worker.py         # Abstract interface for execution workers
│   │   ├── kraken_worker.py       # STUB — teammate implements Kraken CLI calls
│   │   └── aerodrome_worker.py    # STUB — teammate implements web3.py + Risk Router
│   │
│   ├── agents/
│   │   ├── strategy_agent.py      # Trade signals via RAG + indicators + Groq
│   │   ├── risk_manager.py        # 8-rule deterministic gate (no LLM)
│   │   └── auditor_agent.py       # PnL analysis + self-correction + RAG storage
│   │
│   ├── onchain/                   # STUBS — teammate implements
│   │   ├── identity.py            # ERC-8004 Agent Identity (ERC-721)
│   │   ├── reputation.py          # Reputation Registry
│   │   ├── validator.py           # Validation Registry
│   │   ├── signing.py             # EIP-712 typed data signing
│   │   ├── wallet.py              # EIP-1271 smart contract wallet
│   │   └── risk_router.py         # Surge Risk Router submission
│   │
│   ├── rag/
│   │   ├── vector_store.py        # ChromaDB (with sqlite3 patch for Python 3.9)
│   │   └── retriever.py           # Local sentence-transformers embeddings
│   │
│   └── processing/
│       ├── cleaner.py             # Kraken CSV loader + event detection
│       ├── embedder.py            # OHLCV → RAG narrative chunks
│       └── indicators.py          # RSI, EMA, ATR, Volume Z-score, Bollinger %B
│
├── scripts/
│   ├── prepare_data.py            # Copy & validate Kraken OHLCV files
│   ├── bootstrap_memory.py        # Seed RAG with crisis lessons from real data
│   ├── simulator.py               # Paper trading simulator with self-correction
│   ├── register_agent.py          # Mint ERC-8004 Agent Identity NFT
│   └── claim_sandbox_capital.py   # Register with Hackathon Capital Vault
│
├── data/
│   ├── historical/                # XBTUSD_60.csv, ETHUSD_60.csv (copied by prepare_data.py)
│   ├── synthetic/
│   └── chroma_db/                 # ChromaDB persistence (gitignored)
│
├── tests/
│   └── red_team/
│       └── adversarial_agent.py   # 13 attack vectors vs Risk Manager
│
├── docker-compose.yml             # ChromaDB service
├── pyproject.toml
└── README.md
```

---

## The Self-Correction Loop

Every trade triggers a full feedback cycle:

```
1. OBSERVE
   └─ Worker: pull live ticker + OHLCV
   └─ Indicators: compute RSI, EMA, ATR, Volume Z-score, Bollinger %B
   └─ Regime: classify as bullish_breakout / bearish_breakdown / overbought /
              oversold / volume_spike_up / volume_spike_down / sideways / normal
   └─ RAG: "What happened in similar conditions before?"

2. HYPOTHESIZE
   └─ Strategy Agent (Groq llama-3.3-70b) reads:
       - Current indicators + regime
       - RAG: similar historical market cycles
       - RAG: auditor lessons from past similar trades
   └─ Returns: TradeSignal(direction, size_pct, confidence, reasoning)

3. GATE (Risk Manager — no LLM, deterministic)
   └─ Check 1: confidence >= min_signal_confidence
   └─ Check 2: pair in allowed_pairs
   └─ Check 3: balance >= balance_floor_usd
   └─ Check 4: daily_pnl_loss < max_daily_loss_pct
   └─ Check 5: daily_trades < max_trades_per_day
   └─ Check 6: seconds_since_last_trade >= min_trade_interval_seconds
   └─ Check 7: size_pct <= max_position_size_pct (soft cap, not veto)
   └─ Check 8: size_pct >= min_position_size_pct
   └─ EIP-712 TradeIntent constructed and signed (EIP-155 chain-id bound)
   └─ Surge Risk Router validates on-chain: position limits, leverage, whitelist
   └─ Decision: APPROVE → execute, or VETO → log and skip

4. EXECUTE
   └─ CeFi: order placed via Kraken CLI (--sandbox for paper mode)
   └─ DeFi: signed TradeIntent → Aerodrome via Risk Router on Base
   └─ Trade ID + tx hash logged to RAG

5. AUDIT (Self-Correction)
   └─ Auditor Agent (Groq llama-3.3-70b) receives:
       a. Trade record (pair, direction, entry/exit, PnL)
       b. Signal that triggered it (confidence, reasoning)
       c. Current risk_policy.yaml
       d. RAG: similar past lessons
   └─ Output: root_cause, lesson, optional config_update
   └─ If config_update: risk_policy.yaml updated within param_bounds
   └─ Lesson embedded locally and stored in RAG "lessons" collection

6. TRUST SIGNAL (ERC-8004)
   └─ Auditor posts Validation Artifact → ERC-8004 Validation Registry
   └─ PnL delta posted → ERC-8004 Reputation Registry
   └─ Tag: "Strategy Optimized | stop_loss_pct: 2.0→1.5 | tx: 0x..."
```

**Observed self-correction (Aug 2024 simulation):**
- Agent bought BTC/USD at $53,555 (RSI oversold signal at the crash bottom)
- Stop-loss hit → Auditor analysis → `stop_loss_pct: 2.0% → 1.5%` applied automatically

---

## Technical Indicators

Computed fresh on every `MarketData` snapshot before the strategy prompt is built.

| Indicator | Description | Used For |
|-----------|-------------|----------|
| RSI(14) | Relative Strength Index | Overbought/oversold detection |
| EMA(20) | Fast exponential moving average | Short-term trend |
| EMA(50) | Slow exponential moving average | Medium-term trend |
| EMA(200) | Long-term moving average | Major trend bias |
| ATR(14) | Average True Range | Volatility unit for stop sizing |
| Volume Z-score | Std devs above 20-period mean | Abnormal volume detection |
| Bollinger %B | Price position in Bollinger Band | Mean reversion signals |
| Trend EMA20/50 | % gap between EMA20 and EMA50 | Crossover / divergence |
| price_chg_pct | Single-candle % change | Shock event detection |

**Regime classifier** maps indicator combination → one label fed to both the Strategy Agent prompt and the RAG similarity query:

`bullish_breakout` · `bearish_breakdown` · `overbought` · `oversold` · `volume_spike_up` · `volume_spike_down` · `sideways` · `normal`

---

## RAG Memory: Synthetic Experience

The agent's vector database is seeded **before first live trade** using real Kraken historical data.

### Data Sources

| File | Pair | Candles | Date Range |
|------|------|---------|------------|
| `XBTUSD_60.csv` | BTC/USD | 96,381 | 2013–2025 |
| `ETHUSD_60.csv` | ETH/USD | 87,690 | 2015–2025 |
| `XBTUSD_1440.csv` | BTC/USD daily | 4,457 | 2013–2025 |
| `ETHUSD_1440.csv` | ETH/USD daily | 3,794 | 2015–2025 |

### Key Events Covered

| Event | Date | Candles in DB |
|-------|------|---------------|
| LUNA / UST collapse | May 2022 | 721 (BTC) + 721 (ETH) |
| FTX collapse | November 2022 | 697 (BTC) + 697 (ETH) |
| Bitcoin Halving correction | April–May 2024 | 1,436 (BTC) + 1,436 (ETH) |
| Yen carry trade flash crash | August 2024 | 721 (BTC) + 721 (ETH) |

### Bootstrap Pipeline

```
Kraken OHLCV CSV
      │
      ▼
load_ohlcv_df()          ← adds all technical indicators
      │
      ▼
detect_significant_events()   ← multi-type: price_drop, price_spike,
      │                           volume_sell_spike, volume_buy_spike,
      │                           rsi_overbought, rsi_oversold
      ▼ (top events by severity)
_build_lesson_prompt()   ← 48-candle context + indicators + 24-candle outcome
      │
      ▼
Groq llama-3.1-8b-instant    ← 500k tokens/day free tier
      │
      ▼
lesson text              ← "Warning: RSI=72 on low volume. Agent should have..."
      │
      ▼
sentence-transformers    ← all-MiniLM-L6-v2, 384-dim, runs locally
      │
      ▼
ChromaDB "lessons" collection
```

**Lesson prompt format:**

```
At event candle (2024-08-05 04:00 UTC):
  Price: $53,555  |  Change: -8.5%
  RSI(14): 24.1 [oversold]
  EMA20: $58,200  |  EMA50: $61,400  (gap: -5.2% — bearish crossover)
  Volume Z-score: +4.1 [very high]
  Bollinger %B: 0.02
  Regime: volume_spike_down

Outcome 24h later: $49,200 (-8.1% from event)

A trading agent was considering buying BTC/USD at the event candle.
What warning signs were visible? What should it have done?
Name one specific risk_policy.yaml parameter to adjust.
```

---

## Simulation Results

### August 2024 — Yen Carry Trade Flash Crash (BTC/USD)

| Metric | Value |
|--------|-------|
| Windows tested | 12 (50-candle windows across August 2024) |
| Trades placed | 1 (entry at crash bottom, RSI oversold) |
| Vetoed | 11 (HOLD signals — agent recognised uncertain regime) |
| PnL | -$5.00 (-0.05%) — stop-loss triggered |
| Self-corrections | 1: `stop_loss_pct 2.0% → 1.5%` |
| New RAG lessons | 1 |

### Flash Crash Scenario — 9 Worst BTC/USD Single-Candle Drops (All History)

| Metric | Value |
|--------|-------|
| Crashes tested | 9 (worst drops from 2013–2025) |
| Trades placed | 0 |
| Vetoed | 9 (all correctly identified as `volume_spike_down` / `bearish_breakdown`) |
| Capital preserved | 100% |

The agent correctly refuses to trade into all 9 worst historical crashes due to RAG memory of similar past conditions.

---

## Execution Layers: Kraken + Aerodrome

### Kraken CLI (CeFi) — Teammate implements `kraken_worker.py`

```bash
# Install Kraken CLI
git clone https://github.com/krakenfx/kraken-cli.git
cd kraken-cli && cargo build --release
export PATH="$PATH:$(pwd)/target/release"

kraken configure --api-key $KRAKEN_API_KEY --api-secret $KRAKEN_API_SECRET
```

| Operation | CLI Command |
|-----------|-------------|
| Get ticker | `kraken market ticker --pair XBTUSD` |
| Get OHLCV | `kraken market ohlc --pair XBTUSD --interval 1` |
| Market order | `kraken order add --pair XBTUSD --type buy --ordertype market --volume 0.01` |
| Paper mode | `kraken --sandbox order add ...` |
| MCP server | `kraken mcp serve` |

### Aerodrome Finance (DeFi — Base L2) — Teammate implements `aerodrome_worker.py`

```python
# Construct and submit a signed swap TradeIntent
intent = TradeIntent(
    agent_id=AGENT_ID,
    pair="WETH/USDC",
    direction="buy",
    amount=500000000000000000,   # 0.5 ETH in wei
    dex="aerodrome",
    chain_id=8453,
    nonce=get_nonce(),
    deadline=int(time.time()) + 300
)
signed = signing.sign_eip712(intent, wallet)
risk_router.submit(signed)
```

---

## ERC-8004 Trust & Identity Layer

### On-Chain Events Per Trade

| Event | Registry | Data |
|-------|----------|------|
| Agent registered | Identity Registry | Name, wallet, capabilities |
| TradeIntent signed | Local EIP-712 | Pair, size, direction, timestamp |
| Trade executed | Validation Registry | tx hash, PnL delta |
| Risk veto | Validation Registry | Reason, blocked intent hash |
| Config self-updated | Validation Registry | Param, old → new value |
| Reputation updated | Reputation Registry | Cumulative PnL, Sharpe, drawdown |

### Register Agent

```bash
python scripts/register_agent.py
python scripts/claim_sandbox_capital.py
```

---

## EIP-712 Trade Intent Signing

```python
TRADE_INTENT_TYPES = {
    "TradeIntent": [
        {"name": "agentId",   "type": "uint256"},
        {"name": "pair",      "type": "string"},
        {"name": "direction", "type": "string"},   # "buy" | "sell"
        {"name": "amount",    "type": "uint256"},
        {"name": "dex",       "type": "string"},   # "kraken" | "aerodrome"
        {"name": "chainId",   "type": "uint256"},  # EIP-155
        {"name": "nonce",     "type": "uint256"},
        {"name": "deadline",  "type": "uint256"},
    ]
}
```

EIP-1271 support in `wallet.py` allows the agent to operate behind a Safe smart contract wallet.

---

## Risk Router Integration (Surge)

Python `risk_manager.py` pre-validates all rules locally before constructing the EIP-712 signature. The on-chain Surge Risk Router is the final enforcement layer.

| Check | Where enforced |
|-------|---------------|
| Confidence threshold | Python only |
| Pair whitelist | Python + on-chain |
| Balance floor | Python only |
| Daily loss limit | Python + on-chain |
| Max position size | Python + on-chain |
| Max leverage | Python + on-chain |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM — Strategy | Groq `llama-3.3-70b-versatile` |
| LLM — Auditing | Groq `llama-3.3-70b-versatile` |
| LLM — Bootstrap | Groq `llama-3.1-8b-instant` (500k tokens/day free) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| Vector Store | ChromaDB (local, persisted) |
| CeFi Execution | Kraken CLI (Rust binary, MCP server) |
| DeFi Execution | Aerodrome Finance (AMM on Base L2) |
| On-Chain Trust | ERC-8004 — Identity, Reputation, Validation |
| Risk Enforcement | Surge Risk Router contract |
| Signing | EIP-712 typed data + EIP-1271 + EIP-155 |
| Language | Python 3.9+ |
| Dependencies | pyproject.toml (poetry) |
| Network | Base L2 (`chainId: 8453`) / Base Sepolia (`84532`) |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Rust + Cargo (to build Kraken CLI)
- Docker & Docker Compose (optional — for standalone ChromaDB)
- A funded wallet on Base L2 (or Base Sepolia for testnet)
- Kraken account with API keys
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone & Install

```bash
git clone https://github.com/your-org/sentinel-8004.git
cd sentinel-8004
pip install openai pydantic pyyaml python-dotenv chromadb \
            pysqlite3-binary sentence-transformers pandas numpy
```

### 2. Configure Environment

```bash
cp .env.example .env
```

```env
# Required — get free at console.groq.com
GROQ_API_KEY=gsk_...
GROQ_API_KEY_2=gsk_...   # optional second key for rate limit fallback

# Kraken (teammate fills in)
KRAKEN_API_KEY=your_key
KRAKEN_API_SECRET=your_secret

# On-chain (teammate fills in)
AGENT_WALLET_PRIVATE_KEY=0x...
BASE_RPC_URL=https://mainnet.base.org
CHAIN_ID=8453
```

### 3. Prepare Historical Data

```bash
# Copy Kraken OHLCV files to data/historical/ and validate
python scripts/prepare_data.py --raw-dir /path/to/Kraken_OHLCVT/master_q4
```

---

## Running the Pipeline

### Step 1 — Seed RAG Memory

```bash
# Full bootstrap: 379 lessons + 2,918 market cycle chunks
# Uses llama-3.1-8b-instant (500k tokens/day free)
python scripts/bootstrap_memory.py --since 2022-01-01 --max-events 200

# Or skip LLM calls (just index market cycles, no API usage):
python scripts/bootstrap_memory.py --skip-lessons
```

### Step 2 — Simulate & Self-Correct

```bash
# August 2024 yen carry trade flash crash
python scripts/simulator.py --scenario custom_dates \
  --since 2024-08-01 --until 2024-08-31 --pair BTC/USD

# 9 worst historical BTC/USD single-candle drops
python scripts/simulator.py --scenario flash_crash --pair BTC/USD

# 10 low-volatility windows (sideways regime)
python scripts/simulator.py --scenario sideways --pair ETH/USD

# 100 random historical windows — main training run
python scripts/simulator.py --scenario random_N --n 100 --pair BTC/USD --capital 10000

# After simulation, inspect learned parameters:
cat config/risk_policy.yaml
```

### Step 3 — Red Team the Risk Manager

```bash
python tests/red_team/adversarial_agent.py --rounds 50
```

### Step 4 — Go Live (after teammate integrates Kraken + onchain)

```bash
# Paper mode: Kraken CLI --sandbox + testnet DeFi
python -m src.core.orchestrator --mode paper

# Live mode: real Kraken + Base mainnet
python -m src.core.orchestrator --mode live
```

---

## Division of Work

```
SENTINEL-8004
     │
     ├── Agent Layer (Nursena) ← THIS REPO — COMPLETE
     │     ├── Strategy Agent (Groq + RAG + indicators)
     │     ├── Risk Manager (deterministic, 8 rules)
     │     ├── Auditor Agent (Groq + self-correction)
     │     ├── RAG pipeline (ChromaDB + sentence-transformers)
     │     ├── Technical indicators (RSI/EMA/ATR/Volume/BB)
     │     ├── Data preprocessing (Kraken CSV → indicators)
     │     ├── Bootstrap memory (2,918 cycles + 380 lessons)
     │     ├── Simulator (real history + SL/TP + self-correction)
     │     └── Orchestrator (Observe→Gate→Execute→Audit loop)
     │
     └── Execution + Onchain Layer (Teammate) ← TO IMPLEMENT
           ├── KrakenWorker  — implement src/core/kraken_worker.py
           ├── AerodromeWorker — implement src/core/aerodrome_worker.py
           └── Onchain layer — implement src/onchain/*.py
                 ├── signing.py   (EIP-712 — types already defined)
                 ├── identity.py  (ERC-8004 NFT mint)
                 ├── reputation.py (Reputation Registry)
                 ├── validator.py  (Validation Artifacts)
                 ├── risk_router.py (Surge Router submit)
                 └── wallet.py    (EIP-1271)
```

### Interface Contract (what teammate must implement)

```python
class BaseWorker(ABC):
    async def get_market_data(self, pair: str, num_candles: int = 100) -> MarketData: ...
    async def get_portfolio(self) -> PortfolioState: ...
    async def execute_order(self, order: Order) -> ExecutionResult: ...
    async def close_position(self, pair: str, sandbox: bool = True) -> ExecutionResult: ...
```

The `Order` object carries: `pair, direction, size_pct, sandbox, stop_loss_pct, take_profit_pct, execution_layer`.
The `ExecutionResult` must return: `success, trade_id, executed_price, executed_size, execution_layer, tx_hash (DeFi only)`.

---

## Hackathon Track Compliance

### Kraken CLI Track

| Requirement | Implementation |
|-------------|----------------|
| Kraken CLI as execution layer | `src/core/kraken_worker.py` — subprocess + MCP server interface |
| Autonomous AI workflow | Orchestrator: Observe → Gate → Execute → Audit |
| Market data retrieval | `kraken market ticker/ohlc` feeds Strategy Agent |
| Trade execution | `kraken order add` after Risk Manager approval |
| Paper trading | `--sandbox` flag on all simulation runs |
| Read-only API key for leaderboard | Provided at submission |

### ERC-8004 / DeFi Track (Surge)

| Requirement | Implementation |
|-------------|----------------|
| ERC-8004 Agent Identity (ERC-721) | `src/onchain/identity.py` |
| ERC-8004 Reputation Registry | `src/onchain/reputation.py` — PnL signals every trade close |
| ERC-8004 Validation Registry | `src/onchain/validator.py` — Auditor artifacts after self-correction |
| EIP-712 typed data signatures | `src/onchain/signing.py` — types defined, ready to implement |
| EIP-1271 smart contract wallet | `src/onchain/wallet.py` |
| EIP-155 chain-id binding | chain_id=8453 in all TradeIntents |
| DEX execution via Risk Router | `src/onchain/risk_router.py` |
| Sandbox capital via Capital Vault | `scripts/claim_sandbox_capital.py` |

---

## Roadmap

- [x] Core orchestrator loop (Observe → Gate → Execute → Audit)
- [x] Technical indicators (RSI, EMA, ATR, Volume Z-score, Bollinger %B)
- [x] Regime classifier (8 market states)
- [x] Strategy Agent — Groq llama-3.3-70b + RAG + live indicators
- [x] Risk Manager — 8-rule deterministic gate with live policy reload
- [x] Auditor Agent — Groq post-trade analysis + risk_policy.yaml self-update
- [x] RAG layer — ChromaDB + local sentence-transformers (no API key for embeddings)
- [x] SQLite3 fix for Python 3.9 (pysqlite3-binary)
- [x] Data preprocessing — Kraken CSV loader with auto header detection
- [x] Event detection — price_drop, price_spike, volume spikes, RSI extremes
- [x] Bootstrap memory — 2,918 market cycles + 380 lessons from real crisis data
- [x] Simulator — real historical windows, SL/TP simulation, self-correction loop
- [x] Groq key rotation (auto-fallback on rate limits)
- [x] Abstract BaseWorker interface (ready for teammate integration)
- [x] Execution stubs — KrakenWorker, AerodromeWorker (teammate fills in)
- [x] Onchain stubs — identity, reputation, validator, signing, wallet, risk_router
- [x] EIP-712 type definitions (ready for teammate's implementation)
- [x] Red team adversarial testing suite (13 attack vectors)
- [ ] KrakenWorker — real Kraken CLI subprocess calls (teammate)
- [ ] AerodromeWorker — web3.py + Risk Router integration (teammate)
- [ ] EIP-712 signing implementation (teammate)
- [ ] ERC-8004 on-chain registration (teammate)
- [ ] Real-time monitoring dashboard
- [ ] Subgraph / off-chain indexer for agent activity
- [ ] TEE-backed attestations for verifiable execution proofs

---

## Resources

- [Kraken CLI GitHub Repository](https://github.com/krakenfx/kraken-cli)
- [ERC-8004 Specification](https://eips.ethereum.org/EIPS/eip-8004)
- [Aerodrome Finance Documentation](https://aerodrome.finance/docs)
- [Base L2 Documentation](https://docs.base.org)
- [EIP-712 Typed Structured Data Signing](https://eips.ethereum.org/EIPS/eip-712)
- [EIP-1271 Standard Signature Validation](https://eips.ethereum.org/EIPS/eip-1271)
- [Groq Console](https://console.groq.com)
- [sentence-transformers](https://www.sbert.net)

---

## Disclaimer

SENTINEL-8004 is experimental software built for a hackathon. It interacts with live financial markets and blockchain networks. Use paper trading mode and testnet extensively before deploying real capital. The authors accept no responsibility for financial losses.
