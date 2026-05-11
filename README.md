# Crypto Trading Bot

Automated cryptocurrency trading bot for Bybit USDT perpetual markets. The project combines market scanning, multi-timeframe technical analysis, risk controls, trade execution, Telegram monitoring, and local persistence into a single Python application.

This repository is intended as a portfolio project that demonstrates practical backend automation, exchange API integration, stateful trading workflows, and operational safeguards.

## Highlights

- Scans Bybit linear USDT perpetual markets with turnover-based filtering and kline caching.
- Generates trade setups using EMA ribbon, RSI, MACD, ATR, ADX, Bollinger Bands, VWAP, volume, market structure, and orderbook context.
- Supports queued setups, retest/continuation entries, limit order handling, partial take-profits, stop-loss management, and trailing behavior.
- Includes configurable risk controls for daily loss limits, daily trade limits, cooldowns, single-position execution, fixed or percent-based risk, and max position sizing.
- Provides Telegram commands for live bot status, balance, trade history, pause/resume, logs, and controlled shutdown.
- Persists trades and daily stats in SQLite for local reporting and recovery.
- Ships with Docker Compose support and a testnet smoke-run checklist.

## Tech Stack

- Python 3.12
- Bybit API via `pybit`
- pandas, numpy, ta
- SQLite
- python-telegram-bot
- Docker / Docker Compose

## Project Structure

```text
main.py                  # Bot orchestration, execution flow, runtime state
scanner.py               # Bybit symbol discovery, ticker/kline fetching, caching
strategy.py              # Signal analysis and execution trigger logic
risk_manager.py          # Daily limits, cooldowns, position/risk constraints
database.py              # SQLite trade and daily stats persistence
telegram_bot.py          # Telegram command interface
orderbook.py             # Orderbook context scoring
correlation.py           # Correlation-aware trade filtering
indicators.py            # Technical indicator helpers
config.py                # Environment-driven configuration
flow_test_driver.py      # Flow testing helper
backtest.py              # Backtesting utility
Dockerfile               # Container image
docker-compose.yml       # Local container runtime
TESTNET_SMOKE_RUN.md     # Manual testnet validation checklist
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/karabulutyagiz/crypto-bot.git
cd crypto-bot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` with your Bybit and optional Telegram credentials.

Recommended first-run settings:

```env
BYBIT_TESTNET=true
BYBIT_DEMO_TRADING=false
TRADING_MODE=sabah
MAX_DAILY_TRADES=2
MAX_POSITION_SIZE_USDT=100
```

### 4. Run basic checks

```bash
python -m py_compile main.py scanner.py strategy.py risk_manager.py
```

### 5. Start the bot

```bash
python main.py
```

## Docker Usage

```bash
docker compose up -d --build
docker compose logs -f bot
docker compose down
```

Runtime logs and `trades.db` are mounted locally for persistence.

## Telegram Commands

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured, the bot exposes:

```text
/status   Current bot and position status
/balance  Account balance summary
/history  Recent trade history
/logs     Recent logs
/pause    Pause new trading actions
/resume   Resume trading
/stop     Stop the bot
/help     Command list
```

## Safety Notes

- Start on Bybit testnet before any live usage.
- Never commit real API keys, Telegram tokens, databases, or logs.
- `.env`, local logs, SQLite databases, and virtual environments are ignored by Git.
- This project is not financial advice and does not guarantee profitability.

## What This Project Demonstrates

- Building a long-running Python service with exchange API integration.
- Designing a stateful trading execution flow with operational constraints.
- Implementing risk management as first-class application logic.
- Handling market data caching, rate-limit awareness, and symbol filtering.
- Providing observability through structured logs, SQLite history, and Telegram commands.
- Packaging a local service with Docker Compose for repeatable deployment.

## Validation

For a safer end-to-end testnet rollout, follow `TESTNET_SMOKE_RUN.md`.
