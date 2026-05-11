from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

import pandas as pd

from config import EXECUTION_CONFIG, STRATEGY_CONFIG, TIME_CONFIG
from indicators import calculate_adx, calculate_atr, calculate_ema, calculate_rsi, detect_market_structure


def is_time_allowed(hour: int) -> bool:
    return TIME_CONFIG["start_hour"] <= hour < TIME_CONFIG["end_hour"]


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()

    close_s = pd.Series(out["close"])
    high_s = pd.Series(out["high"])
    low_s = pd.Series(out["low"])
    out["ema20"] = calculate_ema(close_s, 20)
    out["ema21"] = calculate_ema(close_s, 21)
    out["ema50"] = calculate_ema(close_s, 50)
    out["ema200"] = calculate_ema(close_s, 200)
    out["rsi"] = calculate_rsi(close_s, 14)
    out["atr"] = calculate_atr(high_s, low_s, close_s, 14)
    adx_s, _, _ = calculate_adx(high_s, low_s, close_s, 14)
    out["adx"] = adx_s
    if "volume" in out.columns:
        vol_s = pd.to_numeric(out["volume"], errors="coerce")
        out["vol_ma"] = vol_s.rolling(window=20).mean()
    else:
        out["vol_ma"] = 0.0
    return out


def _is_btc(symbol: str) -> bool:
    s = (symbol or "").upper()
    return "BTC" in s


def _trend_state_1h(df_1h: pd.DataFrame, symbol: str) -> tuple[str, str, int]:
    if df_1h is None or len(df_1h) < 120:
        return "NONE", "1H veri yetersiz", 0

    d = _with_indicators(df_1h)
    last = d.iloc[-1]

    close = float(last["close"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    if atr <= 0:
        return "NONE", "1H ATR hesaplanamadi", 0

    chop_mult = float(
        STRATEGY_CONFIG.get(
            "btc_chop_atr_mult" if _is_btc(symbol) else "alt_chop_atr_mult",
            0.10 if _is_btc(symbol) else 0.15,
        )
    )
    if abs(close - ema200) < chop_mult * atr:
        return "NONE", "1H chop filtresi", -1

    quality = 0
    if abs(ema50 - ema200) >= 0.5 * atr:
        quality += 1

    if close > ema200 and ema50 > ema200 and rsi > 50:
        return "LONG", "1H trend LONG", 1
    if close < ema200 and ema50 < ema200 and rsi < 50:
        return "SHORT", "1H trend SHORT", 1

    return "NONE", "1H trend izni yok", 0


def _trend_state_30m(df_30m: pd.DataFrame, symbol: str) -> tuple[str, str, float]:
    if df_30m is None or len(df_30m) < 80:
        return "NONE", "30M veri yetersiz", 0.0

    d = _with_indicators(df_30m)
    last = d.iloc[-1]

    close = float(last["close"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    if atr <= 0:
        return "NONE", "30M ATR hesaplanamadi", 0.0

    chop_mult = float(
        STRATEGY_CONFIG.get(
            "btc_30m_chop_atr_mult" if _is_btc(symbol) else "alt_30m_chop_atr_mult",
            0.10 if _is_btc(symbol) else 0.15,
        )
    )
    if abs(close - ema200) < chop_mult * atr:
        return "NONE", "30M chop filtresi", -1

    if abs(close - ema200) < chop_mult * atr:
        if close > ema50:
            return "LONG", "30M chop filtresi", -1.0
        if close < ema50:
            return "SHORT", "30M chop filtresi", -1.0
        return "NONE", "30M chop filtresi", -1.0

    if close > ema50 and ema50 > ema200 and rsi >= 52:
        return "LONG", "30M trend LONG strong", 2.0
    if close > ema50 and rsi >= 50:
        return "LONG", "30M trend LONG weak", 1.0

    if close < ema50 and ema50 < ema200 and rsi <= 48:
        return "SHORT", "30M trend SHORT strong", 2.0
    if close < ema50 and rsi <= 50:
        return "SHORT", "30M trend SHORT weak", 1.0

    return "NONE", "30M trend izni yok", 0.0


def _find_sweep_15m(df_15m: pd.DataFrame, direction: str, symbol: str) -> tuple[Optional[int], str, int]:
    if df_15m is None or len(df_15m) < 60:
        return None, "15m veri yetersiz", 0

    d = _with_indicators(df_15m)
    n = int(STRATEGY_CONFIG.get("sweep_lookback_15m", 16))
    spike_mult = float(
        STRATEGY_CONFIG.get(
            "btc_sweep_spike_atr_mult" if _is_btc(symbol) else "alt_sweep_spike_atr_mult",
            2.8 if _is_btc(symbol) else 3.5,
        )
    )

    idx: Optional[int] = None
    for i in range(n, len(d)):
        row = d.iloc[i]
        atr = float(row["atr"])
        if atr <= 0:
            continue

        candle_range = float(row["high"] - row["low"])
        if candle_range > spike_mult * atr:
            continue

        prev = d.iloc[i - n : i]
        swing_high = float(prev["high"].max())
        swing_low = float(prev["low"].min())

        if direction == "LONG":
            if float(row["high"]) > swing_high and float(row["close"]) < swing_high:
                idx = i
        else:
            if float(row["low"]) < swing_low and float(row["close"]) > swing_low:
                idx = i

    if idx is None:
        return None, "15m sweep yok", 0
    return idx, "15m sweep tespit edildi", 1


def _structure_break_15m(df_15m: pd.DataFrame, sweep_idx: int, direction: str) -> tuple[bool, str, int]:
    d = _with_indicators(df_15m)
    breakout_level = _get_breakout_level_15m(d, sweep_idx, direction)
    if breakout_level is None or breakout_level <= 0:
        return False, "SetupState=NoBreakoutLevel", 0

    if sweep_idx >= len(d) - 1:
        return False, "SetupState=StructureBreakPending", 0

    tail = d.iloc[sweep_idx + 1 :]
    if len(tail) == 0:
        return False, "SetupState=StructureBreakPending", 0

    if direction == "LONG":
        broke = bool((tail["close"] > breakout_level).any())
    else:
        broke = bool((tail["close"] < breakout_level).any())

    if not broke:
        return False, "SetupState=StructureBreakPending", 0
    return True, "SetupState=SweepDetected+StructureBreak", 1


def _get_breakout_level_15m(df_15m: pd.DataFrame, sweep_idx: int, direction: str) -> Optional[float]:
    d = _with_indicators(df_15m)
    if sweep_idx is None or sweep_idx <= 0 or sweep_idx >= len(d):
        return None

    n = int(STRATEGY_CONFIG.get("sweep_lookback_15m", 16))
    start = max(0, sweep_idx - n)
    prev = d.iloc[start:sweep_idx]
    if len(prev) == 0:
        return None

    if direction == "LONG":
        return float(prev["high"].max())
    return float(prev["low"].min())


def _continuation_pullback_metrics(
    d: pd.DataFrame,
    direction: str,
    breakout_level: float,
) -> Dict[str, float]:
    lookback = max(6, int(EXECUTION_CONFIG.get("continuation_impulse_lookback_candles", 16)))
    window = d.iloc[-lookback:] if len(d) > lookback else d
    if window is None or len(window) < 4:
        return {
            "pullback_fib": 0.0,
            "pullback_candles": 0.0,
            "impulse_size": 0.0,
            "pullback_depth": 0.0,
            "pullback_low": 0.0,
            "pullback_high": 0.0,
        }

    if direction == "LONG":
        impulse_rel_pos = int(window["high"].values.argmax())
        impulse_high = float(window["high"].iloc[impulse_rel_pos])
        impulse_start = float(breakout_level) if breakout_level > 0 else float(window["low"].iloc[: impulse_rel_pos + 1].min())
        impulse_size = impulse_high - impulse_start
        retrace_slice = window.iloc[impulse_rel_pos + 1 :]
        retrace_candles = len(retrace_slice)
        if impulse_size <= 0 or retrace_candles <= 0:
            return {
                "pullback_fib": 0.0,
                "pullback_candles": float(max(0, retrace_candles)),
                "impulse_size": max(0.0, impulse_size),
                "pullback_depth": 0.0,
                "pullback_low": 0.0,
                "pullback_high": impulse_high,
            }

        pullback_low = float(retrace_slice["low"].min())
        pullback_depth = max(0.0, impulse_high - pullback_low)
        pullback_fib = pullback_depth / impulse_size
        return {
            "pullback_fib": pullback_fib,
            "pullback_candles": float(retrace_candles),
            "impulse_size": impulse_size,
            "pullback_depth": pullback_depth,
            "pullback_low": pullback_low,
            "pullback_high": impulse_high,
        }

    impulse_rel_pos = int(window["low"].values.argmin())
    impulse_low = float(window["low"].iloc[impulse_rel_pos])
    impulse_start = float(breakout_level) if breakout_level > 0 else float(window["high"].iloc[: impulse_rel_pos + 1].max())
    impulse_size = impulse_start - impulse_low
    retrace_slice = window.iloc[impulse_rel_pos + 1 :]
    retrace_candles = len(retrace_slice)
    if impulse_size <= 0 or retrace_candles <= 0:
        return {
            "pullback_fib": 0.0,
            "pullback_candles": float(max(0, retrace_candles)),
            "impulse_size": max(0.0, impulse_size),
            "pullback_depth": 0.0,
            "pullback_low": impulse_low,
            "pullback_high": 0.0,
        }

    pullback_high = float(retrace_slice["high"].max())
    pullback_depth = max(0.0, pullback_high - impulse_low)
    pullback_fib = pullback_depth / impulse_size
    return {
        "pullback_fib": pullback_fib,
        "pullback_candles": float(retrace_candles),
        "impulse_size": impulse_size,
        "pullback_depth": pullback_depth,
        "pullback_low": impulse_low,
        "pullback_high": pullback_high,
    }


def _nearest_psych_level(price: float) -> float:
    if price <= 0:
        return 0.0
    if price >= 10000:
        step = 1000.0
    elif price >= 1000:
        step = 100.0
    elif price >= 100:
        step = 10.0
    elif price >= 10:
        step = 1.0
    elif price >= 1:
        step = 0.1
    elif price >= 0.1:
        step = 0.01
    else:
        step = 0.001
    return round(round(price / step) * step, 8)


def _psychology_score(direction: str, breakout_level: float, close: float, rsi: float) -> tuple[float, str]:
    if not bool(STRATEGY_CONFIG.get("psychology_enabled", True)):
        return 0.0, "Psych:off"

    score = 0.0
    tags = []
    ref_price = breakout_level if breakout_level > 0 else close
    round_level = _nearest_psych_level(ref_price)
    tol_bps = float(STRATEGY_CONFIG.get("psych_round_level_tolerance_bps", 25)) / 10000.0
    tolerance = max(1e-9, ref_price * tol_bps)

    if round_level > 0 and abs(ref_price - round_level) <= tolerance:
        bonus = float(STRATEGY_CONFIG.get("psych_round_number_bonus", 0.5))
        score += bonus
        tags.append(f"round_bonus@{round_level:.6f}")

    crowded_long = float(STRATEGY_CONFIG.get("psych_rsi_crowded_long", 68))
    crowded_short = float(STRATEGY_CONFIG.get("psych_rsi_crowded_short", 32))
    crowd_penalty = float(STRATEGY_CONFIG.get("psych_rsi_crowded_penalty", 0.6))

    if direction == "LONG" and rsi >= crowded_long:
        score -= crowd_penalty
        tags.append(f"crowded_long_rsi={rsi:.1f}")
    elif direction == "SHORT" and rsi <= crowded_short:
        score -= crowd_penalty
        tags.append(f"crowded_short_rsi={rsi:.1f}")

    if not tags:
        return 0.0, "Psych:neutral"
    return score, "Psych:" + ",".join(tags)


def _psychology_structure_score(df_15m: pd.DataFrame, direction: str, breakout_level: float) -> tuple[float, str]:
    if breakout_level <= 0 or df_15m is None or len(df_15m) < 10:
        return 0.0, "PsychStruct:none"

    d = _with_indicators(df_15m)
    window = d.iloc[-20:] if len(d) > 20 else d
    tol_bps = float(STRATEGY_CONFIG.get("psych_round_level_tolerance_bps", 25)) / 10000.0
    tolerance = max(1e-9, breakout_level * tol_bps)

    if direction == "LONG":
        tests = int(((window["high"] - breakout_level).abs() <= tolerance).sum())
    else:
        tests = int(((window["low"] - breakout_level).abs() <= tolerance).sum())

    if tests >= 2:
        bonus = float(STRATEGY_CONFIG.get("psych_test_count_bonus", 0.4))
        return bonus, f"PsychStruct:tested_{tests}x"
    return 0.0, "PsychStruct:untested"


def _entry_state_5m(
    df_5m: pd.DataFrame,
    direction: str,
    breakout_level: Optional[float] = None,
) -> tuple[bool, bool, str, float, float, Dict[str, float]]:
    if df_5m is None or len(df_5m) < 40:
        return False, False, "5m veri yetersiz", 0.0, 0.0, {}

    d = _with_indicators(df_5m)
    last = d.iloc[-1]

    close = float(last["close"])
    open_price = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])
    ema50 = float(last["ema50"])
    ema21 = float(last["ema21"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    adx = float(last.get("adx", 0) or 0)
    volume = float(last.get("volume", 0) or 0)
    vol_ma = float(last.get("vol_ma", 0) or 0)
    if atr <= 0:
        return False, False, "5m ATR hesaplanamadi", 0.0, 0.0, {}

    volume_filter_enabled = bool(EXECUTION_CONFIG.get("volume_confirmation_enabled", False))
    volume_min_ratio = float(EXECUTION_CONFIG.get("volume_min_ratio", 0.85))
    volume_ratio = (volume / vol_ma) if vol_ma > 0 else 0.0
    volume_ok = (not volume_filter_enabled) or (volume_ratio >= volume_min_ratio)

    adx_filter_enabled = bool(EXECUTION_CONFIG.get("adx_filter_enabled", False))
    adx_min_value = float(EXECUTION_CONFIG.get("adx_min_value", 20))
    adx_ok = (not adx_filter_enabled) or (adx >= adx_min_value)

    market_structure_enabled = bool(EXECUTION_CONFIG.get("market_structure_enabled", False))
    ms_lookback = max(3, int(EXECUTION_CONFIG.get("market_structure_lookback", 5)))
    bull_structure, bear_structure, _ = detect_market_structure(d, lookback=ms_lookback)
    structure_ok = True
    if market_structure_enabled:
        structure_ok = bull_structure if direction == "LONG" else bear_structure

    ema_slope_lookback = max(1, int(EXECUTION_CONFIG.get("continuation_ema_slope_lookback", 3)))
    ema21_prev = float(d["ema21"].iloc[-(ema_slope_lookback + 1)]) if len(d) > ema_slope_lookback else ema21
    ema_alignment_ok = (ema21 >= ema50) if direction == "LONG" else (ema21 <= ema50)
    ema_slope_ok = (ema21 >= ema21_prev) if direction == "LONG" else (ema21 <= ema21_prev)

    if breakout_level is None or breakout_level <= 0:
        return False, False, "EntryState=NoBreakoutLevel", 0.0, 0.0, {}

    zone_band_atr = float(EXECUTION_CONFIG.get("trigger_pullback_band_atr", 0.18)) * atr
    zone_band_bps = (float(EXECUTION_CONFIG.get("pullback_tolerance_bps", 2)) / 10000.0) * close
    retest_band = max(zone_band_atr, zone_band_bps)

    body = max(1e-9, abs(close - open_price))
    candle_range = max(1e-9, high - low)
    body_ratio = body / candle_range
    lower_wick = max(0.0, min(open_price, close) - low)
    upper_wick = max(0.0, high - max(open_price, close))
    close_location = (close - low) / candle_range
    min_wick_body = float(EXECUTION_CONFIG.get("trigger_confirm_body_ratio", 0.18))
    impulse_atr = abs(close - breakout_level) / max(1e-9, atr)
    pullback_metrics = _continuation_pullback_metrics(d, direction, float(breakout_level))
    pullback_fib = float(pullback_metrics.get("pullback_fib", 0.0) or 0.0)
    pullback_candles = int(pullback_metrics.get("pullback_candles", 0) or 0)
    min_pullback_fib = float(EXECUTION_CONFIG.get("continuation_min_pullback_fib", 0.382))
    min_pullback_candles = max(1, int(EXECUTION_CONFIG.get("continuation_min_pullback_candles", 2)))
    pullback_fib_ok = pullback_fib >= min_pullback_fib
    pullback_candles_ok = pullback_candles >= min_pullback_candles
    rsi_long_max = float(EXECUTION_CONFIG.get("continuation_rsi_long_max", 65))
    rsi_short_min = float(EXECUTION_CONFIG.get("continuation_rsi_short_min", 35))

    support_lookback = max(6, int(EXECUTION_CONFIG.get("support_lookback_candles", 24)))
    support_window = d.iloc[-support_lookback:] if len(d) > support_lookback else d
    recent_support = float(cast(Any, support_window["low"].min())) if len(support_window) > 0 else float(low)
    support_touch_atr = float(EXECUTION_CONFIG.get("support_touch_atr", 0.20))
    support_touched = low <= (recent_support + (support_touch_atr * atr))
    recent_swing_high = float(cast(Any, support_window["high"].max())) if len(support_window) > 0 else float(high)
    drop_size_atr = max(0.0, (recent_swing_high - close) / max(1e-9, atr))

    in_zone = abs(close - breakout_level) <= (retest_band * 2.0) or (
        low <= (breakout_level + retest_band) and high >= (breakout_level - retest_band)
    )

    short_pullback_retest_required = False
    ema_retest_ok = False
    short_rejection_ok = False
    reclaim_required = bool(EXECUTION_CONFIG.get("trigger_require_reclaim", True))
    rejection_required = bool(EXECUTION_CONFIG.get("trigger_require_rejection", True))
    trigger_close_location_min = float(EXECUTION_CONFIG.get("trigger_min_close_location", 0.55))
    breakout_hold_atr = float(EXECUTION_CONFIG.get("trigger_breakout_hold_atr", 0.05))
    continuation_close_location_min = float(EXECUTION_CONFIG.get("continuation_min_close_location", 0.60))
    continuation_breakout_hold_atr = float(EXECUTION_CONFIG.get("continuation_breakout_hold_atr", 0.10))

    if direction == "LONG":
        cond_close = close > (breakout_level + (breakout_hold_atr * atr))
        cond_rsi = rsi >= float(EXECUTION_CONFIG.get("retest_rsi_long_min", 48))
        rsi_extreme_ok = rsi <= rsi_long_max
        near_retest = low <= (breakout_level + retest_band) and close >= (breakout_level - retest_band)
        reclaim = close > breakout_level and close > open_price
        rejection = lower_wick >= (body * min_wick_body)
        close_location_ok = close_location >= trigger_close_location_min
        retest_ema_alignment_ok = close >= ema21 and ema_alignment_ok
        retest_ema_slope_ok = ema_slope_ok

        cont_rsi_ok = rsi >= float(EXECUTION_CONFIG.get("continuation_rsi_long_min", 55))
        cont_distance_ok = impulse_atr <= float(EXECUTION_CONFIG.get("continuation_max_distance_atr", 1.25))
        cont_wick_ok = lower_wick <= (body * float(EXECUTION_CONFIG.get("continuation_max_against_wick_ratio", 0.60)))
        cont_breakout_hold_ok = close > (breakout_level + (continuation_breakout_hold_atr * atr))
        cont_close_location_ok = close_location >= continuation_close_location_min
        continuation_ok = (
            bool(EXECUTION_CONFIG.get("allow_continuation_entry", True))
            and cond_close
            and cont_rsi_ok
            and cont_distance_ok
            and cont_wick_ok
            and cont_breakout_hold_ok
            and cont_close_location_ok
            and close > open_price
            and body_ratio >= float(EXECUTION_CONFIG.get("continuation_min_body_ratio", 0.55))
            and impulse_atr >= float(EXECUTION_CONFIG.get("continuation_impulse_min_atr", 0.90))
            and impulse_atr <= float(EXECUTION_CONFIG.get("continuation_impulse_max_atr", 2.20))
            and pullback_fib_ok
            and pullback_candles_ok
            and rsi_extreme_ok
            and (not bool(EXECUTION_CONFIG.get("continuation_require_ema_alignment", True)) or ema_alignment_ok)
            and (not bool(EXECUTION_CONFIG.get("continuation_require_ema_slope", True)) or ema_slope_ok)
        )

        bounce_enabled = bool(EXECUTION_CONFIG.get("long_support_bounce_enabled", False))
        bounce_rsi_max = float(EXECUTION_CONFIG.get("long_support_bounce_rsi_max", 35))
        bounce_wick_body = float(EXECUTION_CONFIG.get("long_support_bounce_min_wick_body_ratio", 1.30))
        bounce_min_drop_atr = float(EXECUTION_CONFIG.get("long_support_bounce_min_drop_atr", 1.60))
        bounce_triggered = (
            bounce_enabled
            and support_touched
            and (lower_wick >= (body * bounce_wick_body))
            and rsi <= bounce_rsi_max
            and close > open_price
            and drop_size_atr >= bounce_min_drop_atr
        )
    else:
        cond_close = close < (breakout_level - (breakout_hold_atr * atr))
        cond_rsi = rsi <= float(EXECUTION_CONFIG.get("retest_rsi_short_max", 52))
        rsi_extreme_ok = rsi >= rsi_short_min
        near_retest = high >= (breakout_level - retest_band) and close <= (breakout_level + retest_band)
        reclaim = close < breakout_level and close < open_price
        rejection = upper_wick >= (body * min_wick_body)
        close_location_ok = (1.0 - close_location) >= trigger_close_location_min
        retest_ema_alignment_ok = close <= ema21 and ema_alignment_ok
        retest_ema_slope_ok = ema_slope_ok
        cont_rsi_ok = rsi <= float(EXECUTION_CONFIG.get("continuation_rsi_short_max", 45))
        cont_distance_ok = impulse_atr <= float(EXECUTION_CONFIG.get("continuation_max_distance_atr", 1.25))
        cont_wick_ok = upper_wick <= (body * float(EXECUTION_CONFIG.get("continuation_max_against_wick_ratio", 0.60)))
        cont_breakout_hold_ok = close < (breakout_level - (continuation_breakout_hold_atr * atr))
        cont_close_location_ok = (1.0 - close_location) >= continuation_close_location_min
        continuation_ok = (
            bool(EXECUTION_CONFIG.get("allow_continuation_entry", True))
            and cond_close
            and cont_rsi_ok
            and cont_distance_ok
            and cont_wick_ok
            and cont_breakout_hold_ok
            and cont_close_location_ok
            and close < open_price
            and body_ratio >= float(EXECUTION_CONFIG.get("continuation_min_body_ratio", 0.55))
            and impulse_atr >= float(EXECUTION_CONFIG.get("continuation_impulse_min_atr", 0.90))
            and impulse_atr <= float(EXECUTION_CONFIG.get("continuation_impulse_max_atr", 2.20))
            and pullback_fib_ok
            and pullback_candles_ok
            and rsi_extreme_ok
            and (not bool(EXECUTION_CONFIG.get("continuation_require_ema_alignment", True)) or ema_alignment_ok)
            and (not bool(EXECUTION_CONFIG.get("continuation_require_ema_slope", True)) or ema_slope_ok)
        )

        short_support_guard_enabled = bool(EXECUTION_CONFIG.get("short_support_guard_enabled", False))
        short_support_guard_distance_atr = float(EXECUTION_CONFIG.get("short_support_guard_distance_atr", 0.60))
        short_support_guard_drop_atr = float(EXECUTION_CONFIG.get("short_support_guard_drop_atr", 2.20))
        near_support = (close - recent_support) <= (short_support_guard_distance_atr * atr)
        short_liquidity_trap = (
            short_support_guard_enabled
            and near_support
            and support_touched
            and drop_size_atr >= short_support_guard_drop_atr
        )

        short_pullback_retest_required = bool(EXECUTION_CONFIG.get("short_pullback_retest_required", False))
        short_pullback_retest_band_atr = float(EXECUTION_CONFIG.get("short_pullback_retest_band_atr", 0.28))
        short_rejection_wick_body_ratio = float(EXECUTION_CONFIG.get("short_rejection_wick_body_ratio", 1.10))
        ema_retest_ok = (
            abs(close - ema21) <= (short_pullback_retest_band_atr * atr)
            or abs(close - ema50) <= (short_pullback_retest_band_atr * atr)
            or high >= ema21
            or high >= ema50
        )
        short_rejection_ok = upper_wick >= (body * short_rejection_wick_body_ratio) and close < open_price

        if short_pullback_retest_required:
            continuation_ok = continuation_ok and ema_retest_ok and short_rejection_ok

        if short_liquidity_trap:
            return (
                in_zone,
                False,
                (
                    "EntryState=ShortBlockedLiquiditySweep "
                    f"support={recent_support:.6f} drop_atr={drop_size_atr:.2f} "
                    f"near_support={int(near_support)} support_touch={int(support_touched)}"
                ),
                0.0,
                impulse_atr,
                {
                    "recent_support": recent_support,
                    "drop_size_atr": drop_size_atr,
                    "near_support": float(int(near_support)),
                    "support_touch": float(int(support_touched)),
                },
            )

        bounce_triggered = False

    reclaim_ok = reclaim or (not reclaim_required)
    rejection_ok = rejection or (not rejection_required)
    close_required = bool(EXECUTION_CONFIG.get("trigger_require_close", True))
    rsi_required = bool(EXECUTION_CONFIG.get("trigger_require_rsi", True))
    near_retest_required = bool(EXECUTION_CONFIG.get("trigger_require_near_retest", True))
    retest_ema_alignment_required = bool(EXECUTION_CONFIG.get("trigger_require_ema_alignment", True))
    retest_ema_slope_required = bool(EXECUTION_CONFIG.get("trigger_require_ema_slope", False))
    close_ok = cond_close or (not close_required)
    rsi_ok = cond_rsi or (not rsi_required)
    near_retest_ok = near_retest or (not near_retest_required)
    retest_ema_alignment_gate = retest_ema_alignment_ok or (not retest_ema_alignment_required)
    retest_ema_slope_gate = retest_ema_slope_ok or (not retest_ema_slope_required)
    retest_triggered = (
        in_zone
        and near_retest_ok
        and reclaim_ok
        and rejection_ok
        and close_ok
        and rsi_ok
        and rsi_extreme_ok
        and close_location_ok
        and retest_ema_alignment_gate
        and retest_ema_slope_gate
    )
    if direction == "SHORT" and short_pullback_retest_required:
        retest_triggered = retest_triggered and ema_retest_ok and short_rejection_ok
    triggered = (retest_triggered or continuation_ok or bounce_triggered) and volume_ok and adx_ok and structure_ok
    entry_quality = 0.0
    if cond_close:
        entry_quality += 0.5
    if near_retest:
        entry_quality += 0.5
    if reclaim:
        entry_quality += 0.5
    if rejection:
        entry_quality += 0.5
    if continuation_ok:
        entry_quality += 0.5
    if cond_rsi:
        entry_quality += 0.5
    if close_location_ok:
        entry_quality += 0.5

    if retest_triggered:
        base_entry_score = float(STRATEGY_CONFIG.get("retest_entry_score", 3.0))
    elif continuation_ok:
        base_entry_score = float(STRATEGY_CONFIG.get("continuation_entry_score", 2.2))
    else:
        base_entry_score = 1.0 if in_zone else 0.0
    total_entry_score = base_entry_score + entry_quality
    details = {
        "close_cond": float(int(cond_close)),
        "near_retest": float(int(near_retest)),
        "reclaim": float(int(reclaim)),
        "rejection": float(int(rejection)),
        "close_location_ok": float(int(close_location_ok)),
        "close_location": close_location,
        "cont_ok": float(int(continuation_ok)),
        "cont_distance_ok": float(int(cont_distance_ok)),
        "cont_wick_ok": float(int(cont_wick_ok)),
        "cont_breakout_hold_ok": float(int(cont_breakout_hold_ok)),
        "cont_close_location_ok": float(int(cont_close_location_ok)),
        "pullback_fib": pullback_fib,
        "pullback_fib_ok": float(int(pullback_fib_ok)),
        "pullback_candles": float(pullback_candles),
        "pullback_candles_ok": float(int(pullback_candles_ok)),
        "rsi_extreme_ok": float(int(rsi_extreme_ok)),
        "rsi": rsi,
        "retest_low": float(d["low"].iloc[-max(1, int(EXECUTION_CONFIG.get("trigger_pullback_lookback_candles", 2))) :].min()),
        "retest_high": float(d["high"].iloc[-max(1, int(EXECUTION_CONFIG.get("trigger_pullback_lookback_candles", 2))) :].max()),
        "pullback_low": float(pullback_metrics.get("pullback_low", 0.0) or 0.0),
        "pullback_high": float(pullback_metrics.get("pullback_high", 0.0) or 0.0),
        "rsi_cond": float(int(cond_rsi)),
        "volume_ratio": volume_ratio,
        "volume_ok": float(int(volume_ok)),
        "adx": adx,
        "adx_ok": float(int(adx_ok)),
        "structure_ok": float(int(structure_ok)),
        "ema_alignment_ok": float(int(ema_alignment_ok)),
        "ema_slope_ok": float(int(ema_slope_ok)),
        "retest_ema_alignment_ok": float(int(retest_ema_alignment_ok)),
        "retest_ema_slope_ok": float(int(retest_ema_slope_ok)),
        "entry_quality": entry_quality,
        "recent_support": recent_support,
        "support_touch": float(int(support_touched)),
        "drop_size_atr": drop_size_atr,
    }

    if retest_triggered:
        return True, True, (
            "EntryState=InZone+Triggered "
            f"near_retest=1 reclaim=1 rejection=1 close_loc_ok={int(close_location_ok)} "
            f"ema_align_ok={int(retest_ema_alignment_ok)} breakout={breakout_level:.6f}"
        ), total_entry_score, impulse_atr, details
    if continuation_ok:
        return in_zone, True, (
            "EntryState=Continuation+Triggered "
            f"impulse_atr={impulse_atr:.2f} body_ratio={body_ratio:.2f} cont_distance_ok={int(cont_distance_ok)} "
            f"cont_wick_ok={int(cont_wick_ok)} pullback_fib={pullback_fib:.3f} "
            f"pullback_candles={pullback_candles} rsi_extreme_ok={int(rsi_extreme_ok)} cont_hold_ok={int(cont_breakout_hold_ok)} "
            f"cont_close_loc_ok={int(cont_close_location_ok)} "
            f"ema_align_ok={int(ema_alignment_ok)} ema_slope_ok={int(ema_slope_ok)} breakout={breakout_level:.6f}"
        ), total_entry_score, impulse_atr, details
    if bounce_triggered:
        return True, True, (
            "EntryState=SupportBounce+Triggered "
            f"support={recent_support:.6f} rsi={rsi:.2f} lower_wick={lower_wick:.6f} drop_atr={drop_size_atr:.2f}"
        ), total_entry_score, impulse_atr, details
    if in_zone:
        return (
            True,
            False,
            f"EntryState=InZone+NoTrigger close_cond={int(cond_close)} rsi_cond={int(cond_rsi)} near_retest={int(near_retest)} "
            f"reclaim={int(reclaim)} rejection={int(rejection)} cont_ok={int(continuation_ok)} impulse_atr={impulse_atr:.2f} "
            f"pullback_fib={pullback_fib:.3f} pullback_candles={pullback_candles} rsi_extreme_ok={int(rsi_extreme_ok)} "
            f"cont_hold_ok={int(cont_breakout_hold_ok)} cont_close_loc_ok={int(cont_close_location_ok)} "
            f"volume_ok={int(volume_ok)} adx_ok={int(adx_ok)} structure_ok={int(structure_ok)} "
            f"ema_align_ok={int(ema_alignment_ok)} ema_slope_ok={int(ema_slope_ok)} close_loc_ok={int(close_location_ok)} "
            f"close={close:.6f} breakout={breakout_level:.6f} ema50={ema50:.6f} rsi={rsi:.2f}",
            total_entry_score,
            impulse_atr,
            details,
        )
    return (
        False,
        False,
        f"EntryState=OutOfZone close_cond={int(cond_close)} rsi_cond={int(cond_rsi)} near_retest={int(near_retest)} "
        f"reclaim={int(reclaim)} rejection={int(rejection)} cont_ok={int(continuation_ok)} impulse_atr={impulse_atr:.2f} "
        f"pullback_fib={pullback_fib:.3f} pullback_candles={pullback_candles} rsi_extreme_ok={int(rsi_extreme_ok)} "
        f"cont_hold_ok={int(cont_breakout_hold_ok)} cont_close_loc_ok={int(cont_close_location_ok)} "
        f"volume_ok={int(volume_ok)} adx_ok={int(adx_ok)} structure_ok={int(structure_ok)} "
        f"ema_align_ok={int(ema_alignment_ok)} ema_slope_ok={int(ema_slope_ok)} close_loc_ok={int(close_location_ok)} "
        f"close={close:.6f} breakout={breakout_level:.6f} ema50={ema50:.6f} rsi={rsi:.2f}",
        total_entry_score,
        impulse_atr,
        details,
    )


def _build_levels(df_15m: pd.DataFrame, direction: str, symbol: str) -> Optional[Dict[str, Any]]:
    d = _with_indicators(df_15m)
    if len(d) < 30:
        return None

    last = d.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr"])
    if pd.isna(close) or pd.isna(atr) or close <= 0 or atr <= 0:
        return None

    recent = d.iloc[-20:]
    sl_buf = float(STRATEGY_CONFIG.get("sl_buffer_atr", 0.25)) * atr

    if direction == "LONG":
        swing_low = float(recent["low"].min())
        sl_trigger = swing_low - sl_buf
        sl_limit = sl_trigger - sl_buf
        risk = close - sl_trigger
    else:
        swing_high = float(recent["high"].max())
        sl_trigger = swing_high + sl_buf
        sl_limit = sl_trigger + sl_buf
        risk = sl_trigger - close

    if pd.isna(sl_trigger) or pd.isna(risk) or risk <= 0:
        return None

    if _is_btc(symbol):
        min_mult = float(STRATEGY_CONFIG.get("btc_sl_min_atr", 0.45))
        max_mult = float(STRATEGY_CONFIG.get("btc_sl_max_atr", 2.1))
        risk_pct = 0.75
    else:
        min_mult = float(STRATEGY_CONFIG.get("alt_sl_min_atr", 0.60))
        max_mult = float(STRATEGY_CONFIG.get("alt_sl_max_atr", 2.6))
        risk_pct = 0.5

    if risk < min_mult * atr or risk > max_mult * atr:
        return None

    tp1_rr = float(STRATEGY_CONFIG.get("tp1_rr", 1.0))
    tp2_rr = float(STRATEGY_CONFIG.get("tp2_rr", 2.0))
    tp3_rr = float(STRATEGY_CONFIG.get("tp3_rr", 3.0))

    tp1_w = float(STRATEGY_CONFIG.get("tp1_percent", 40)) / 100.0
    tp2_w = float(STRATEGY_CONFIG.get("tp2_percent", 40)) / 100.0
    tp3_w = float(STRATEGY_CONFIG.get("tp3_percent", 20)) / 100.0
    w_sum = max(1e-9, tp1_w + tp2_w + tp3_w)
    blended_rr = (tp1_rr * tp1_w + tp2_rr * tp2_w + tp3_rr * tp3_w) / w_sum

    if direction == "LONG":
        tp = close + risk * tp1_rr
        tp2 = close + risk * tp2_rr
        tp3 = close + risk * tp3_rr
    else:
        tp = close - risk * tp1_rr
        tp2 = close - risk * tp2_rr
        tp3 = close - risk * tp3_rr

    if any(pd.isna(v) for v in (tp, tp2, tp3, sl_trigger, sl_limit)):
        return None

    return {
        "entry": close,
        "stop_loss": sl_trigger,
        "sl_trigger": sl_trigger,
        "sl_limit": sl_limit,
        "tp1": tp,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr": blended_rr,
        "atr": atr,
        "tp1_percent": float(STRATEGY_CONFIG.get("tp1_percent", 40)),
        "tp2_percent": float(STRATEGY_CONFIG.get("tp2_percent", 40)),
        "tp3_percent": float(STRATEGY_CONFIG.get("tp3_percent", 20)),
        "risk_percent_override": risk_pct,
    }


def _build_levels_atr_fallback(df_5m: pd.DataFrame, direction: str, symbol: str) -> Optional[Dict[str, Any]]:
    d = _with_indicators(df_5m)
    if len(d) < 30:
        return None

    last = d.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr"])
    if pd.isna(close) or pd.isna(atr) or close <= 0 or atr <= 0:
        return None

    if _is_btc(symbol):
        min_mult = float(STRATEGY_CONFIG.get("btc_sl_min_atr", 0.45))
        max_mult = float(STRATEGY_CONFIG.get("btc_sl_max_atr", 2.1))
        risk_pct = 0.75
    else:
        min_mult = float(STRATEGY_CONFIG.get("alt_sl_min_atr", 0.60))
        max_mult = float(STRATEGY_CONFIG.get("alt_sl_max_atr", 2.6))
        risk_pct = 0.5

    sl_mult = min(max(1.2, min_mult), max_mult)
    risk = sl_mult * atr
    if risk <= 0:
        return None

    if direction == "LONG":
        sl_trigger = close - risk
        sl_limit = sl_trigger - (0.2 * atr)
    else:
        sl_trigger = close + risk
        sl_limit = sl_trigger + (0.2 * atr)

    tp1_rr = float(STRATEGY_CONFIG.get("tp1_rr", 1.0))
    tp2_rr = float(STRATEGY_CONFIG.get("tp2_rr", 2.0))
    tp3_rr = float(STRATEGY_CONFIG.get("tp3_rr", 3.0))

    tp1_w = float(STRATEGY_CONFIG.get("tp1_percent", 40)) / 100.0
    tp2_w = float(STRATEGY_CONFIG.get("tp2_percent", 40)) / 100.0
    tp3_w = float(STRATEGY_CONFIG.get("tp3_percent", 20)) / 100.0
    w_sum = max(1e-9, tp1_w + tp2_w + tp3_w)
    blended_rr = (tp1_rr * tp1_w + tp2_rr * tp2_w + tp3_rr * tp3_w) / w_sum

    if direction == "LONG":
        tp = close + risk * tp1_rr
        tp2 = close + risk * tp2_rr
        tp3 = close + risk * tp3_rr
    else:
        tp = close - risk * tp1_rr
        tp2 = close - risk * tp2_rr
        tp3 = close - risk * tp3_rr

    return {
        "entry": close,
        "stop_loss": sl_trigger,
        "sl_trigger": sl_trigger,
        "sl_limit": sl_limit,
        "tp1": tp,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr": blended_rr,
        "atr": atr,
        "tp1_percent": float(STRATEGY_CONFIG.get("tp1_percent", 40)),
        "tp2_percent": float(STRATEGY_CONFIG.get("tp2_percent", 40)),
        "tp3_percent": float(STRATEGY_CONFIG.get("tp3_percent", 20)),
        "risk_percent_override": risk_pct,
    }


def get_execution_trigger(df: pd.DataFrame, signal: str, breakout_level: Optional[float] = None) -> Tuple[bool, str]:
    direction = (signal or "").upper()
    if direction not in ("LONG", "SHORT"):
        return False, "Yön geçersiz"
    in_zone, triggered, reason, _, _, _ = _entry_state_5m(df, direction, breakout_level=breakout_level)
    if triggered:
        return True, reason
    return False, reason


def analyze(
    trigger_df: pd.DataFrame,
    current_time: tuple,
    bias_df: Optional[pd.DataFrame] = None,
    mtf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    mtf_enabled: bool = True,
    min_mtf_confirmations: int = 2,
    symbol: str = "",
) -> Optional[Dict[str, Any]]:
    del mtf_enabled, min_mtf_confirmations, bias_df

    hour, _, _, _ = current_time
    if not is_time_allowed(hour):
        return {"signal": None, "reason": "İzin verilen saatler dışında"}

    df_5m = trigger_df
    df_15m = (mtf_dfs or {}).get("15") if mtf_dfs else None
    df_30m = (mtf_dfs or {}).get("30") if mtf_dfs else None
    df_1h = (mtf_dfs or {}).get("60") if mtf_dfs else None

    if df_5m is None or len(df_5m) < 60:
        return {"signal": None, "reason": "5m veri yetersiz"}
    if df_15m is None or len(df_15m) < 60:
        return {"signal": None, "reason": "15m veri yetersiz"}

    total_score = 0.0
    reasons = []
    trend_votes = {"LONG": 0, "SHORT": 0}

    trend_30m_state, trend_30m_reason, _score_30m = "NONE", "30M veri yok", 0.0
    if df_30m is not None and len(df_30m) >= 80:
        trend_30m_state, trend_30m_reason, _score_30m = _trend_state_30m(df_30m, symbol)
        if trend_30m_state != "NONE":
            trend_votes[trend_30m_state] += 2
    reasons.append(f"30M:{trend_30m_state}")

    trend_1h_state, trend_1h_reason, _score_1h = "NONE", "1H veri yok", 0
    if df_1h is not None and len(df_1h) >= 120:
        trend_1h_state, trend_1h_reason, _score_1h = _trend_state_1h(df_1h, symbol)
        if trend_1h_state != "NONE":
            trend_votes[trend_1h_state] += 3
    reasons.append(f"1H:{trend_1h_state}")

    direction = None
    if trend_votes["LONG"] > trend_votes["SHORT"]:
        direction = "LONG"
    elif trend_votes["SHORT"] > trend_votes["LONG"]:
        direction = "SHORT"
    elif trend_30m_state != "NONE":
        direction = trend_30m_state

    for test_dir in ["LONG", "SHORT"]:
        sweep_idx_test, sweep_reason_test, score_sweep_test = _find_sweep_15m(df_15m, test_dir, symbol)
        if sweep_idx_test is not None:
            if direction is None:
                direction = test_dir
            break

    if direction is None:
        direction = "LONG"

    htf_consensus = [state for state in (trend_30m_state, trend_1h_state) if state in ("LONG", "SHORT")]
    htf_opposition_count = sum(1 for state in htf_consensus if state != direction)
    full_htf_countertrend = len(htf_consensus) >= 2 and htf_opposition_count == len(htf_consensus)
    if full_htf_countertrend and bool(STRATEGY_CONFIG.get("block_against_full_htf_consensus", True)):
        reasons.append(f"HTF:block_countertrend({direction})")
        return {
            "signal": None,
            "reason": f"HTF tam ters konsensus | {' | '.join(reasons)}",
            "trend_state": trend_30m_state,
            "setup_state": "BlockedHTFConsensus",
            "entry_state": "Pending",
            "trigger_reason": f"30M={trend_30m_state}, 1H={trend_1h_state}, direction={direction}",
            "breakout_level": None,
            "total_score": 0.0,
            "risk_multiplier": 0.75,
        }

    sweep_idx, sweep_reason, _score_sweep = _find_sweep_15m(df_15m, direction, symbol)
    if sweep_idx is not None:
        reasons.append(f"15M:sweep_{direction}")
    else:
        reasons.append("15M:no_sweep")

    structure_ok = False
    structure_reason = ""
    _score_structure = 0
    if sweep_idx is not None:
        structure_ok, structure_reason, _score_structure = _structure_break_15m(df_15m, sweep_idx, direction)
        if structure_ok:
            reasons.append("15M:structure_break")
        else:
            reasons.append("15M:structure_break_pending")

    breakout_level = _get_breakout_level_15m(df_15m, sweep_idx, direction) if sweep_idx is not None else None

    in_zone, triggered, entry_reason, _score_entry, impulse_atr, entry_details = _entry_state_5m(
        df_5m,
        direction,
        breakout_level=breakout_level,
    )
    if in_zone or triggered:
        entry_state = "InZone+Triggered" if triggered else "InZone"
        reasons.append(f"5M:{entry_state}")
    else:
        reasons.append("5M:OutOfZone")

    flag_30m = trend_30m_state != "NONE"
    flag_15m = sweep_idx is not None and structure_ok
    flag_5m = in_zone or triggered

    trend_score = _score_30m
    if trend_1h_state == direction:
        trend_score += float(STRATEGY_CONFIG.get("htf_alignment_bonus", 0.8))
        reasons.append("1H:aligned")
    elif trend_1h_state in ("LONG", "SHORT"):
        trend_score -= float(STRATEGY_CONFIG.get("htf_countertrend_penalty", 1.0))
        reasons.append("1H:countertrend_penalty")

    if trend_30m_state == direction and trend_1h_state == direction:
        trend_score += 0.2
        reasons.append("HTF:dual_alignment")
    setup_score = 0.0
    if sweep_idx is not None and structure_ok:
        setup_score = 2.0
    elif sweep_idx is not None:
        setup_score = 1.0

    ema_alignment_score = 0.0
    d15 = _with_indicators(df_15m)
    if len(d15) > 0:
        last15 = d15.iloc[-1]
        ema20_15 = float(last15["ema20"])
        ema50_15 = float(last15["ema50"])
        ema200_15 = float(last15["ema200"])
        if direction == "LONG" and ema20_15 > ema50_15 > ema200_15:
            ema_alignment_score = 1.0
        if direction == "SHORT" and ema20_15 < ema50_15 < ema200_15:
            ema_alignment_score = 1.0

    rsi_momentum_score = 0.0
    d5 = _with_indicators(df_5m)
    last_rsi5 = 50.0
    if len(d5) > 0:
        rsi5 = float(d5.iloc[-1]["rsi"])
        last_rsi5 = rsi5
        if direction == "LONG" and rsi5 >= 52:
            rsi_momentum_score = 1.0
        if direction == "SHORT" and rsi5 <= 48:
            rsi_momentum_score = 1.0

    impulse_score = 1.0 if impulse_atr >= float(STRATEGY_CONFIG.get("impulse_score_min_atr", 1.5)) else 0.0
    psych_score, psych_reason = _psychology_score(direction, float(breakout_level or 0), float(d5.iloc[-1]["close"]) if len(d5) > 0 else 0.0, last_rsi5)
    psych_struct_score, psych_struct_reason = _psychology_structure_score(df_15m, direction, float(breakout_level or 0))

    total_score = trend_score + setup_score + _score_entry + ema_alignment_score + rsi_momentum_score + impulse_score + psych_score + psych_struct_score

    reasons.append(
        f"ScoreParts:trend={trend_score:.1f},setup={setup_score:.1f},entry={_score_entry:.1f},"
        f"ema={ema_alignment_score:.1f},rsi={rsi_momentum_score:.1f},impulse={impulse_score:.1f},psych={psych_score + psych_struct_score:.1f}"
    )
    reasons.append(psych_reason)
    reasons.append(psych_struct_reason)

    reasons.append(f"TFFlags:30M={int(flag_30m)},15M={int(flag_15m)},5M={int(flag_5m)}")

    risk_multiplier = 1.0
    if trend_30m_state == "NONE" or trend_30m_state != direction:
        risk_multiplier = 0.75

    reasons.append(f"RiskMult:{risk_multiplier}")

    levels = None
    level_source = "none"
    if sweep_idx is not None and structure_ok:
        levels = _build_levels(df_15m, direction, symbol)
        if levels:
            level_source = "15m"
            reasons.append("SL:OK(15M)")
        else:
            reasons.append("SL:fail(15M)")

    if levels is None and df_30m is not None and len(df_30m) >= 80:
        levels = _build_levels(df_30m, direction, symbol)
        if levels:
            level_source = "30m"
            total_score += float(STRATEGY_CONFIG.get("levels_fallback_penalty_30m", -0.5))
            reasons.append("SL:OK(30M-fallback)")
        else:
            reasons.append("SL:fail(30M-fallback)")

    if levels is None:
        levels = _build_levels_atr_fallback(df_5m, direction, symbol)
        if levels:
            level_source = "atr"
            total_score += float(STRATEGY_CONFIG.get("levels_fallback_penalty_atr", -1.0))
            reasons.append("SL:OK(ATR-fallback)")
        else:
            reasons.append("SL:fail(ATR-fallback)")

    if not levels:
        return {
            "signal": None,
            "reason": f"SL hesaplanamadi | {' | '.join(reasons)}",
            "trend_state": trend_30m_state,
            "setup_state": "SweepDetected+StructureBreak" if structure_ok else "Invalid",
            "entry_state": "InZone+Triggered" if triggered else ("InZone" if in_zone else "OutOfZone"),
            "trigger_reason": entry_reason,
            "breakout_level": breakout_level,
            "total_score": total_score,
            "risk_multiplier": risk_multiplier,
        }

    try:
        if bool(STRATEGY_CONFIG.get("use_retest_buffer_sl", True)):
            entry_price = float(levels.get("entry", 0) or 0)
            atr_val = float(levels.get("atr", 0) or 0)
            if entry_price > 0 and atr_val > 0:
                continuation_triggered = "Continuation+Triggered" in str(entry_reason)
                sl_buffer_atr = float(STRATEGY_CONFIG.get("retest_sl_buffer_atr", 0.5))
                sl_limit_buffer_atr = float(STRATEGY_CONFIG.get("retest_sl_limit_buffer_atr", 0.2))
                orig_gap = abs(float(levels.get("sl_limit", levels.get("sl_trigger", 0)) or 0) - float(levels.get("sl_trigger", 0) or 0))
                sl_gap = max(orig_gap, sl_limit_buffer_atr * atr_val)

                if direction == "LONG":
                    anchor_low = float(entry_details.get("pullback_low", 0.0) or 0.0) if continuation_triggered else float(entry_details.get("retest_low", 0.0) or 0.0)
                    if anchor_low <= 0:
                        anchor_low = float(entry_details.get("retest_low", 0.0) or 0.0)
                    candidate_sl = anchor_low - (sl_buffer_atr * atr_val)
                    if 0 < candidate_sl < entry_price:
                        levels["sl_trigger"] = candidate_sl
                        levels["sl_limit"] = candidate_sl - sl_gap
                        levels["stop_loss"] = levels["sl_trigger"]
                        levels["risk"] = entry_price - levels["sl_trigger"]
                        levels["tp1"] = entry_price + (levels["risk"] * float(STRATEGY_CONFIG.get("tp1_rr", 1.0)))
                        levels["tp2"] = entry_price + (levels["risk"] * float(STRATEGY_CONFIG.get("tp2_rr", 2.0)))
                        levels["tp3"] = entry_price + (levels["risk"] * float(STRATEGY_CONFIG.get("tp3_rr", 3.0)))
                        reasons.append("SL:retest_buffer_applied")
                else:
                    anchor_high = float(entry_details.get("pullback_high", 0.0) or 0.0) if continuation_triggered else float(entry_details.get("retest_high", 0.0) or 0.0)
                    if anchor_high <= 0:
                        anchor_high = float(entry_details.get("retest_high", 0.0) or 0.0)
                    candidate_sl = anchor_high + (sl_buffer_atr * atr_val)
                    if candidate_sl > entry_price:
                        levels["sl_trigger"] = candidate_sl
                        levels["sl_limit"] = candidate_sl + sl_gap
                        levels["stop_loss"] = levels["sl_trigger"]
                        levels["risk"] = levels["sl_trigger"] - entry_price
                        levels["tp1"] = entry_price - (levels["risk"] * float(STRATEGY_CONFIG.get("tp1_rr", 1.0)))
                        levels["tp2"] = entry_price - (levels["risk"] * float(STRATEGY_CONFIG.get("tp2_rr", 2.0)))
                        levels["tp3"] = entry_price - (levels["risk"] * float(STRATEGY_CONFIG.get("tp3_rr", 3.0)))
                        reasons.append("SL:retest_buffer_applied")
    except Exception:
        pass

    rr_hard_min = float(STRATEGY_CONFIG.get("rr_hard_min", 1.0))
    rr_soft_min = float(STRATEGY_CONFIG.get("min_rr", 1.2))
    rr_now = float(levels.get("rr", 0) or 0)
    if rr_now < rr_hard_min:
        return {
            "signal": None,
            "reason": f"RR çok düşük ({rr_now:.2f}<{rr_hard_min:.2f}) | {' | '.join(reasons)}",
            "trend_state": trend_30m_state,
            "setup_state": "InvalidRR",
            "entry_state": "InZone+Triggered" if triggered else ("InZone" if in_zone else "OutOfZone"),
            "trigger_reason": entry_reason,
            "breakout_level": breakout_level,
            "total_score": total_score,
            "risk_multiplier": risk_multiplier,
        }
    if rr_now < rr_soft_min:
        total_score += float(STRATEGY_CONFIG.get("rr_soft_penalty", -1.0))
        reasons.append(f"RR:soft_penalty({rr_now:.2f}<{rr_soft_min:.2f})")

    max_queue_impulse = float(EXECUTION_CONFIG.get("max_queue_impulse_atr", 3.5))
    if (not triggered) and (not in_zone) and impulse_atr > max_queue_impulse:
        total_score += float(STRATEGY_CONFIG.get("impulse_far_penalty", -1.0))
        reasons.append(f"5M:far_penalty({impulse_atr:.2f}ATR>{max_queue_impulse:.2f}ATR)")

    min_total_score = float(STRATEGY_CONFIG.get("min_total_score", 4.0))
    if total_score < min_total_score:
        fail_reason = f"Yetersiz puan ({total_score:.1f}<{min_total_score:.1f})"
        return {
            "signal": None,
            "reason": f"{fail_reason} | {' | '.join(reasons)}",
            "trend_state": trend_30m_state,
            "setup_state": "SweepDetected+StructureBreak" if structure_ok else (structure_reason or "SweepDetected+StructureBreakPending"),
            "entry_state": "InZone+Triggered" if triggered else ("InZone" if in_zone else "OutOfZone"),
            "trigger_reason": entry_reason,
            "breakout_level": breakout_level,
            "total_score": total_score,
            "risk_multiplier": risk_multiplier,
        }

    if not triggered:
        return {
            "signal": direction,
            "reason": f"5M entry trigger bekleniyor (QUEUED) | {' | '.join(reasons)}",
            "trend_state": trend_30m_state,
            "setup_state": "QUEUED",
            "entry_state": "InZone" if in_zone else "OutOfZone",
            "trigger_reason": entry_reason,
            "breakout_level": breakout_level,
            "level_source": level_source,
            **levels,
            "quality_score": total_score,
            "total_score": total_score,
            "risk_multiplier": risk_multiplier,
        }

    final_reason = f"{' | '.join(reasons)} | total_score={total_score}"
    return {
        "signal": direction,
        "reason": final_reason,
        "level_source": level_source,
        **levels,
        "trend_strength": "strong" if total_score >= float(STRATEGY_CONFIG.get("strong_total_score", 5.5)) else "medium",
        "weak_mode": False,
        "weak_reason": "",
        "regime": "trend",
        "trend_state": trend_30m_state,
        "setup_state": "TRIGGERED",
        "entry_state": "TRIGGERED",
        "trigger_reason": entry_reason,
        "breakout_level": breakout_level,
        "quality_score": total_score,
        "setup_timeout_minutes": 90,
        "mtf_confirmed": True,
        "mtf_confirmations": 0,
        "mtf_total": 0,
        "total_score": total_score,
        "risk_multiplier": risk_multiplier,
    }
