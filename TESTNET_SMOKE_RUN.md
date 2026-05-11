# Testnet Smoke Run Checklist (Full Universe + Real SL)

## Recommended .env Preset (sabah)

Use this profile for safer first rollout on testnet with full USDT perpetual universe:

```env
BYBIT_TESTNET=true
TRADING_MODE=sabah
FAST_TEST_MODE=false

SCAN_ALL_LINEAR_PERPS=true
SCAN_INTERVAL_SECONDS=20

ENTRY_ORDER_TYPE=limit
PULLBACK_TOLERANCE_BPS=2

POSITION_SIZE_MODE=risk_based
RISK_MODE=percent
LEVERAGE=3
RISK_PERCENT_PER_TRADE=0.8
MAX_POSITION_SIZE_USDT=100
MAX_SL_USDT=40

MAX_DAILY_TRADES=2
DAILY_MAX_LOSSES=2
DAILY_PROFIT_TARGET_PERCENT=2.0
DAILY_LOSS_LIMIT_PERCENT=2.0
MIN_TIME_BETWEEN_TRADES_MINUTES=10
```

Notes:
- `TRADING_MODE=sabah` uses the current baseline profile.
- `TRADING_MODE=gece` keeps the same risk limits but loosens only entry filters to allow a bit more trade flow.
- Orderbook is soft-score only; it does not hard-block trade execution.
- Bot keeps single-position behavior.

## Pre-Run Checks

1) Compile critical modules:

```bash
python -m py_compile main.py scanner.py strategy.py risk_manager.py
```

2) Validate important runtime flags:

```bash
python - <<'PY'
from config import BYBIT_TESTNET, ACTIVE_TRADING_MODE, SCAN_CONFIG, EXECUTION_CONFIG
print('BYBIT_TESTNET =', BYBIT_TESTNET)
print('ACTIVE_TRADING_MODE =', ACTIVE_TRADING_MODE)
print('scan_all_linear_perps =', SCAN_CONFIG.get('scan_all_linear_perps'))
print('entry_order_type =', EXECUTION_CONFIG.get('entry_order_type'))
print('volume_confirmation_enabled =', EXECUTION_CONFIG.get('volume_confirmation_enabled'))
print('adx_filter_enabled =', EXECUTION_CONFIG.get('adx_filter_enabled'))
print('market_structure_enabled =', EXECUTION_CONFIG.get('market_structure_enabled'))
PY
```

3) Verify Telegram commands respond: `/status`, `/balance`, `/history`.

## Live Smoke Procedure (Testnet)

1) Start bot:

```bash
python main.py
```

2) Let it run for at least 3 scan cycles.

3) Watch logs for:
- `setup kuyru` / `5m trigger`
- `sinyal elendi`
- `korelasyon filtresi`
- `pozisyon acildi`

4) If a position opens, verify real SL order exists (triggered reduce-only stop-limit):

```bash
python - <<'PY'
from pybit.unified_trading import HTTP
from config import BYBIT_TESTNET, BYBIT_DEMO_TRADING, BYBIT_API_KEY, BYBIT_API_SECRET

s = HTTP(testnet=BYBIT_TESTNET, demo=BYBIT_DEMO_TRADING, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
resp = s.get_open_orders(category='linear')
orders = resp.get('result', {}).get('list', [])
sl_like = [o for o in orders if str(o.get('reduceOnly')).lower() == 'true' and float(o.get('triggerPrice', 0) or 0) > 0]
print('open_orders=', len(orders))
print('sl_candidate_orders=', len(sl_like))
for o in sl_like[:10]:
    print(o.get('symbol'), o.get('side'), 'trigger=', o.get('triggerPrice'), 'limit=', o.get('price'))
PY
```

5) Check TP/SL life-cycle:
- TP1 partial fill should move SL to BE logic.
- TP2 fill should switch to candle-close trailing updates.
- On close, DB trade status should become `closed`.

## Pass Criteria

- Bot runs without restart loop.
- No qty/tick/min-notional validation errors.
- ADX/volume/market-structure filters appear in reasons and affect trigger decisions.
- Real SL is represented by reduce-only trigger order(s), not only in-memory state.
- Single-position rule is preserved while scanning full universe.

## Rollback

If setup flow is too strict:
1) set `TRADING_MODE=gece`
2) keep `SCAN_ALL_LINEAR_PERPS=true`
3) rerun 1-2 hour smoke on testnet before any live changes
