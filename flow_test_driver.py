import time
from pybit.unified_trading import HTTP

from config import (
    BYBIT_TESTNET,
    BYBIT_DEMO_TRADING,
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    FLOW_TEST_SYMBOL,
)
from scanner import Scanner


def get_live_position(session: HTTP, symbol: str) -> dict | None:
    resp = session.get_positions(category="linear", symbol=symbol)
    for row in resp.get("result", {}).get("list", []):
        size = float(row.get("size", 0) or 0)
        if size > 0:
            return {
                "side": row.get("side", "Buy"),
                "size": size,
            }
    return None


def reduce_market(session: HTTP, symbol: str, side: str, qty: float) -> None:
    if qty <= 0:
        return
    session.place_order(
        category="linear",
        symbol=symbol,
        side=side,
        orderType="Market",
        qty=str(qty),
        reduceOnly=True,
    )


def main() -> None:
    session = HTTP(
        testnet=BYBIT_TESTNET,
        demo=BYBIT_DEMO_TRADING,
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
    )
    scanner = Scanner()
    symbol = FLOW_TEST_SYMBOL

    print(f"[FLOW_TEST] waiting for open position on {symbol}...")
    position = None
    for _ in range(180):
        position = get_live_position(session, symbol)
        if position:
            break
        time.sleep(2)

    if not position:
        print("[FLOW_TEST] no position opened within timeout")
        return

    original_size = float(position["size"])
    close_side = "Sell" if position["side"] == "Buy" else "Buy"
    print(f"[FLOW_TEST] position detected size={original_size}, side={position['side']}")

    tp1_qty = scanner.normalize_qty(symbol, original_size * 0.50)
    print(f"[FLOW_TEST] sending TP1 simulation close qty={tp1_qty}")
    reduce_market(session, symbol, close_side, tp1_qty)

    print("[FLOW_TEST] waiting 70s for bot BE update...")
    time.sleep(70)

    live = get_live_position(session, symbol)
    if not live:
        print("[FLOW_TEST] position already closed after TP1 simulation")
        return

    tp2_target_remaining = original_size * 0.10
    tp2_qty = scanner.normalize_qty(symbol, max(0.0, float(live["size"]) - tp2_target_remaining))
    print(f"[FLOW_TEST] sending TP2 simulation close qty={tp2_qty}")
    reduce_market(session, symbol, close_side, tp2_qty)

    print("[FLOW_TEST] done. check bot logs for TP1/TP2/SL updates")


if __name__ == "__main__":
    main()
