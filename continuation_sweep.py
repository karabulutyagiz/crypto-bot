from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
from pybit.unified_trading import HTTP

from config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_DEMO_TRADING,
    BYBIT_TESTNET,
    EXECUTION_CONFIG,
    STRATEGY_CONFIG,
)
from strategy import analyze, get_execution_trigger


PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "aggressive": {
        "min_total_score": 3.8,
        "min_entry_score": 3.8,
        "min_rr": 1.22,
        "volume_min_ratio": 0.75,
        "adx_min_value": 16,
        "entry_rsi_long_max": 70,
        "entry_rsi_short_min": 36,
    },
    "balanced": {
        "min_total_score": 4.2,
        "min_entry_score": 4.0,
        "min_rr": 1.30,
        "volume_min_ratio": 0.85,
        "adx_min_value": 18,
        "entry_rsi_long_max": 68,
        "entry_rsi_short_min": 38,
    },
    "conservative": {
        "min_total_score": 4.8,
        "min_entry_score": 4.4,
        "min_rr": 1.40,
        "volume_min_ratio": 0.95,
        "adx_min_value": 22,
        "entry_rsi_long_max": 64,
        "entry_rsi_short_min": 40,
    },
}


@dataclass
class OpenTrade:
    side: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    tp1_w: float
    tp2_w: float
    tp3_w: float
    timestamp: Any
    initial_risk: float
    entry_type: str
    tp1_hit: bool = False
    tp2_hit: bool = False
    be_after_tp1: bool = False
    realized_r: float = 0.0


def _fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    session = HTTP(
        testnet=BYBIT_TESTNET,
        demo=BYBIT_DEMO_TRADING,
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
    )

    now_dt = datetime.now(UTC)
    now_ms = int(now_dt.timestamp() * 1000)
    start_ms = int((now_dt - timedelta(days=days)).timestamp() * 1000)
    cursor = start_ms
    rows: List[list] = []

    while cursor < now_ms:
        resp = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=cursor,
            end=now_ms,
            limit=1000,
        )
        data = resp.get("result", {}).get("list", [])
        if not data:
            break

        batch = sorted(data, key=lambda x: int(x[0]))
        rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

        if len(batch) < 1000:
            break

    if not rows:
        raise RuntimeError("Kline verisi çekilemedi")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rules = {
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "60": "60min",
    }
    rule = rules.get(tf)
    if not rule:
        raise ValueError(f"Desteklenmeyen timeframe: {tf}")
    out = (
        df.set_index("timestamp")
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
        .reset_index()
    )
    return out


def _build_trade_from_result(result: Dict[str, Any], now_ts: Any, fallback_entry: float) -> OpenTrade | None:
    side = str(result.get("signal", ""))
    entry = float(result.get("entry", fallback_entry) or 0.0)
    stop = float(result.get("sl_trigger", result.get("stop_loss", 0.0)) or 0.0)
    tp1 = float(result.get("tp1", 0.0) or 0.0)
    tp2 = float(result.get("tp2", 0.0) or 0.0)
    tp3 = float(result.get("tp3", 0.0) or 0.0)

    if entry <= 0 or stop <= 0 or tp1 <= 0 or tp2 <= 0 or tp3 <= 0:
        return None
    if abs(entry - stop) <= 1e-9:
        return None
    if side == "LONG" and not (stop < entry < tp1 < tp2 < tp3):
        return None
    if side == "SHORT" and not (stop > entry > tp1 > tp2 > tp3):
        return None

    tp1_w = float(result.get("tp1_percent", STRATEGY_CONFIG.get("tp1_percent", 40))) / 100.0
    tp2_w = float(result.get("tp2_percent", STRATEGY_CONFIG.get("tp2_percent", 35))) / 100.0
    tp3_w = float(result.get("tp3_percent", STRATEGY_CONFIG.get("tp3_percent", 25))) / 100.0
    w_sum = max(1e-9, tp1_w + tp2_w + tp3_w)

    return OpenTrade(
        side=side,
        entry=entry,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        tp1_w=tp1_w / w_sum,
        tp2_w=tp2_w / w_sum,
        tp3_w=tp3_w / w_sum,
        timestamp=now_ts,
        initial_risk=abs(entry - stop),
        entry_type="instant" if str(result.get("setup_state", "")) == "TRIGGERED" else "queued",
    )


def _close_trade(trades: List[dict], t: OpenTrade, ts: Any, reason: str) -> None:
    trades.append(
        {
            "entry_time": t.timestamp,
            "exit_time": ts,
            "side": t.side,
            "pnl_r": t.realized_r,
            "reason": reason,
            "tp1_hit": t.tp1_hit,
            "tp2_hit": t.tp2_hit,
            "be_after_tp1": t.be_after_tp1,
            "entry_type": t.entry_type,
        }
    )


def _simulate_exit(candle: pd.Series, trade: OpenTrade, trades: List[dict]) -> Tuple[bool, OpenTrade | None]:
    high = float(cast(Any, candle["high"]))
    low = float(cast(Any, candle["low"]))
    ts = pd.Timestamp(cast(Any, candle["timestamp"]))
    if pd.isna(ts):
        return False, trade

    if trade.side == "LONG":
        if not trade.tp1_hit and low <= trade.stop:
            trade.realized_r -= 1.0
            _close_trade(trades, trade, ts, "SL")
            return True, None

        if not trade.tp1_hit and high >= trade.tp1:
            trade.tp1_hit = True
            trade.realized_r += trade.tp1_w * ((trade.tp1 - trade.entry) / trade.initial_risk)
            trade.stop = trade.entry

        if trade.tp1_hit and not trade.tp2_hit and high >= trade.tp2:
            trade.tp2_hit = True
            trade.realized_r += trade.tp2_w * ((trade.tp2 - trade.entry) / trade.initial_risk)

        if high >= trade.tp3:
            trade.realized_r += trade.tp3_w * ((trade.tp3 - trade.entry) / trade.initial_risk)
            _close_trade(trades, trade, ts, "TP3")
            return True, None

        if trade.tp1_hit and low <= trade.stop:
            trade.be_after_tp1 = True
            _close_trade(trades, trade, ts, "BE_AFTER_TP1")
            return True, None

    else:
        if not trade.tp1_hit and high >= trade.stop:
            trade.realized_r -= 1.0
            _close_trade(trades, trade, ts, "SL")
            return True, None

        if not trade.tp1_hit and low <= trade.tp1:
            trade.tp1_hit = True
            trade.realized_r += trade.tp1_w * ((trade.entry - trade.tp1) / trade.initial_risk)
            trade.stop = trade.entry

        if trade.tp1_hit and not trade.tp2_hit and low <= trade.tp2:
            trade.tp2_hit = True
            trade.realized_r += trade.tp2_w * ((trade.entry - trade.tp2) / trade.initial_risk)

        if low <= trade.tp3:
            trade.realized_r += trade.tp3_w * ((trade.entry - trade.tp3) / trade.initial_risk)
            _close_trade(trades, trade, ts, "TP3")
            return True, None

        if trade.tp1_hit and high >= trade.stop:
            trade.be_after_tp1 = True
            _close_trade(trades, trade, ts, "BE_AFTER_TP1")
            return True, None

    return False, trade


def _calc_max_loss_streak(trades: List[dict]) -> int:
    streak = 0
    max_streak = 0
    for t in trades:
        if float(t.get("pnl_r", 0.0)) < 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak


def _calc_max_dd_r(trades: List[dict]) -> float:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        eq += float(t.get("pnl_r", 0.0))
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_backtest(df_1m: pd.DataFrame, params: Dict[str, float]) -> Dict[str, float]:
    backup_exec = dict(EXECUTION_CONFIG)
    backup_strategy = dict(STRATEGY_CONFIG)
    EXECUTION_CONFIG.update(
        {
            "volume_min_ratio": float(params["volume_min_ratio"]),
            "adx_min_value": float(params["adx_min_value"]),
            "entry_rsi_long_max": float(params["entry_rsi_long_max"]),
            "entry_rsi_short_min": float(params["entry_rsi_short_min"]),
            "min_entry_score": float(params["min_entry_score"]),
        }
    )
    STRATEGY_CONFIG.update(
        {
            "min_rr": float(params["min_rr"]),
            "min_total_score": float(params["min_total_score"]),
        }
    )

    try:
        df_5m = _resample_ohlcv(df_1m, "5")
        df_15m = _resample_ohlcv(df_1m, "15")

        open_trade: OpenTrade | None = None
        pending_setup: Dict[str, Any] | None = None
        trades: List[dict] = []

        for i in range(180, len(df_1m)):
            candle = df_1m.iloc[i]
            now_ts = pd.Timestamp(cast(Any, candle["timestamp"]))
            if pd.isna(now_ts):
                continue

            if open_trade is not None:
                closed, open_trade = _simulate_exit(candle, open_trade, trades)
                if closed:
                    continue

            if open_trade is not None:
                continue

            d1 = pd.DataFrame(df_1m[df_1m["timestamp"] <= now_ts].copy())
            d5 = pd.DataFrame(df_5m[df_5m["timestamp"] <= now_ts].copy())
            d15 = pd.DataFrame(df_15m[df_15m["timestamp"] <= now_ts].copy())

            if len(d1) < 60 or len(d5) < 60 or len(d15) < 60:
                continue

            current_time = (now_ts.hour, now_ts.minute, now_ts.day, now_ts.weekday())

            if pending_setup is not None:
                expires_at = pd.Timestamp(cast(Any, pending_setup.get("expires_at")))
                if not pd.isna(expires_at) and now_ts > expires_at:
                    pending_setup = None
                else:
                    triggered, _trigger_reason = get_execution_trigger(
                        d1,
                        str(pending_setup.get("signal", "")),
                        breakout_level=pending_setup.get("breakout_level"),
                        mtf_dfs={"5": d5, "15": d15},
                        current_time=current_time,
                        symbol="BACKTEST",
                    )
                    if triggered:
                        refreshed = analyze(
                            d1,
                            current_time,
                            mtf_dfs={"5": d5, "15": d15},
                            symbol="BACKTEST",
                        )
                        if refreshed and str(refreshed.get("setup_state", "")) == "TRIGGERED":
                            trade = _build_trade_from_result(refreshed, now_ts, float(candle["close"]))
                            if trade is not None:
                                open_trade = trade
                                pending_setup = None
                                continue
                        pending_setup = None

            result = analyze(
                d1,
                current_time,
                mtf_dfs={"5": d5, "15": d15},
                symbol="BACKTEST",
            )
            if not result or result.get("signal") is None:
                continue

            setup_state = str(result.get("setup_state", ""))
            if setup_state == "QUEUED":
                timeout_minutes = int(result.get("setup_timeout_minutes", 30) or 30)
                pending_setup = {
                    "signal": result.get("signal"),
                    "breakout_level": result.get("breakout_level"),
                    "expires_at": now_ts + pd.Timedelta(minutes=timeout_minutes),
                }
                continue
            if setup_state != "TRIGGERED":
                continue

            trade = _build_trade_from_result(result, now_ts, float(candle["close"]))
            if trade is not None:
                open_trade = trade

        if open_trade is not None:
            forced_ts = pd.Timestamp(cast(Any, df_1m.iloc[-1]["timestamp"]))
            if not pd.isna(forced_ts):
                _close_trade(trades, open_trade, forced_ts, "FORCED_CLOSE")

        if not trades:
            return {
                "trades": 0,
                "trades_per_day": 0.0,
                "winrate": 0.0,
                "avg_r": 0.0,
                "max_dd_r": 0.0,
                "tp1_hit_rate": 0.0,
                "be_after_tp1_count": 0,
                "longest_loss_streak": 0,
            }

        window_start = pd.Timestamp(cast(Any, df_1m.iloc[0]["timestamp"]))
        window_end = pd.Timestamp(cast(Any, df_1m.iloc[-1]["timestamp"]))
        days = max(1.0, (window_end - window_start).total_seconds() / 86400.0)
        wins = [t for t in trades if float(t["pnl_r"]) > 0]
        tp1_hits = [t for t in trades if bool(t.get("tp1_hit"))]
        be_after_tp1 = [t for t in trades if bool(t.get("be_after_tp1"))]
        be_zero_r = [t for t in trades if bool(t.get("be_after_tp1")) and abs(float(t.get("pnl_r", 0.0))) < 1e-9]
        instant_entries = [t for t in trades if str(t.get("entry_type", "")) == "instant"]
        queued_entries = [t for t in trades if str(t.get("entry_type", "")) == "queued"]

        return {
            "trades": len(trades),
            "trades_per_day": len(trades) / days,
            "winrate": (len(wins) / len(trades)) * 100.0,
            "avg_r": sum(float(t["pnl_r"]) for t in trades) / len(trades),
            "max_dd_r": _calc_max_dd_r(trades),
            "tp1_hit_rate": (len(tp1_hits) / len(trades)) * 100.0,
            "be_after_tp1_count": len(be_after_tp1),
            "be_after_tp1_rate": (len(be_after_tp1) / len(trades)) * 100.0,
            "be_zero_r_count": len(be_zero_r),
            "longest_loss_streak": _calc_max_loss_streak(trades),
            "instant_count": len(instant_entries),
            "queued_count": len(queued_entries),
        }
    finally:
        EXECUTION_CONFIG.clear()
        EXECUTION_CONFIG.update(backup_exec)
        STRATEGY_CONFIG.clear()
        STRATEGY_CONFIG.update(backup_strategy)


def _fmt_row(name: str, row: Dict[str, float]) -> str:
    return (
        f"{name:28} | trades={int(row['trades']):4d} | day={row['trades_per_day']:.2f} | "
        f"win={row['winrate']:.1f}% | avgR={row['avg_r']:.3f} | maxDD={row['max_dd_r']:.2f}R | "
        f"tp1={row['tp1_hit_rate']:.1f}% | be_after_tp1={int(row.get('be_after_tp1_count', 0))} "
        f"({row.get('be_after_tp1_rate', 0.0):.1f}%) | loss_streak={int(row['longest_loss_streak'])}"
        f" | be_0R={int(row.get('be_zero_r_count', 0))}"
        f" | instant={int(row.get('instant_count', 0))}, queued={int(row.get('queued_count', 0))}"
    )


def run_sweep(symbol: str, days: int, include_impulse_grid: bool) -> None:
    df_1m = _fetch_klines(symbol=symbol, interval="1", days=days)

    score_values = [3.8, 4.2, 4.6]
    entry_values = [3.8, 4.0, 4.4]
    rr_values = [1.22, 1.30, 1.38] if include_impulse_grid else [float(STRATEGY_CONFIG.get("min_rr", 1.30))]

    rows: List[Tuple[str, Dict[str, float]]] = []
    for min_total, min_entry, min_rr in itertools.product(score_values, entry_values, rr_values):
        params = {
            "min_total_score": min_total,
            "min_entry_score": min_entry,
            "min_rr": min_rr,
            "volume_min_ratio": float(EXECUTION_CONFIG.get("volume_min_ratio", 0.85)),
            "adx_min_value": float(EXECUTION_CONFIG.get("adx_min_value", 20)),
            "entry_rsi_long_max": float(EXECUTION_CONFIG.get("entry_rsi_long_max", 68)),
            "entry_rsi_short_min": float(EXECUTION_CONFIG.get("entry_rsi_short_min", 38)),
        }
        res = run_backtest(df_1m, params)
        label = f"score={min_total}, entry={min_entry}, rr={min_rr}"
        rows.append((label, res))

    rows.sort(key=lambda x: (x[1]["avg_r"], -x[1]["max_dd_r"], x[1]["trades_per_day"]), reverse=True)

    print("\n=== Live-Strategy Sweep Results (sorted by avgR, then lower DD) ===")
    for label, row in rows:
        print(_fmt_row(label, row))


def run_profiles(symbol: str, days: int) -> None:
    df_1m = _fetch_klines(symbol=symbol, interval="1", days=days)

    print("\n=== Profile Comparison ===")
    for profile, p in PROFILE_PRESETS.items():
        res = run_backtest(df_1m, p)
        print(_fmt_row(profile, res))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live strategy parameter sweep backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Bybit linear symbol")
    parser.add_argument("--days", type=int, default=14, help="Lookback days")
    parser.add_argument("--mode", choices=["sweep", "profiles", "both"], default="both")
    parser.add_argument("--include-impulse-grid", action="store_true", help="Use 3x3x3 grid")
    args = parser.parse_args()

    if args.mode in ("sweep", "both"):
        run_sweep(symbol=args.symbol, days=args.days, include_impulse_grid=args.include_impulse_grid)
    if args.mode in ("profiles", "both"):
        run_profiles(symbol=args.symbol, days=args.days)


if __name__ == "__main__":
    main()
