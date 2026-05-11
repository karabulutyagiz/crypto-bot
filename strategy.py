from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from config import EXECUTION_CONFIG, STRATEGY_CONFIG, TIME_CONFIG
from indicators import (
    calculate_atr, calculate_ema, calculate_rsi, calculate_vwap, calculate_cvd,
    detect_ema_ribbon_alignment, calculate_adx, calculate_macd, calculate_bollinger_bands
)


def is_time_allowed(hour: int) -> bool:
    return TIME_CONFIG["start_hour"] <= hour < TIME_CONFIG["end_hour"]


def _closed_candle_view(df: Optional[pd.DataFrame], min_rows: int) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if bool(getattr(df, "attrs", {}).get("closed_candle_view", False)):
        return df.copy() if len(df) >= min_rows else None
    if len(df) <= min_rows:
        return None
    out = df.iloc[:-1].copy()
    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["closed_candle_view"] = True
    return out if len(out) >= min_rows else None


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Temel indikatörleri hesaplar"""
    out = df.copy()
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()

    close_s = pd.Series(out["close"])
    high_s = pd.Series(out["high"])
    low_s = pd.Series(out["low"])
    
    # EMA Ribbon - Scalp için optimize edilmiş
    out["ema8"] = calculate_ema(close_s, 8)
    out["ema7"] = calculate_ema(close_s, 7)
    out["ema13"] = calculate_ema(close_s, 13)
    out["ema21"] = calculate_ema(close_s, 21)
    out["ema55"] = calculate_ema(close_s, 55)
    out["ema200"] = calculate_ema(close_s, 200)
    
    # RSI - Hızlı momentum için RSI9
    out["rsi9"] = calculate_rsi(close_s, 9)
    out["rsi14"] = calculate_rsi(close_s, 14)

    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(close_s, 12, 26, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    
    # ATR
    out["atr"] = calculate_atr(high_s, low_s, close_s, 14)
    
    # ADX
    adx_s, adx_pos, adx_neg = calculate_adx(high_s, low_s, close_s, 14)
    out["adx"] = adx_s
    out["adx_pos"] = adx_pos
    out["adx_neg"] = adx_neg
    
    # Volume based indicators
    if "volume" in out.columns:
        vol_s = pd.Series(pd.to_numeric(out["volume"], errors="coerce"), index=out.index).fillna(0.0)
        out["vol_ma"] = vol_s.rolling(window=20).mean()
        out["vwap"] = calculate_vwap(high_s, low_s, close_s, vol_s)
        out["cvd"] = calculate_cvd(close_s, vol_s)
    else:
        out["vol_ma"] = 0.0
        out["vwap"] = close_s
        out["cvd"] = 0.0

    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close_s, 20, 2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    
    return out


def _higher_low_intact(df: pd.DataFrame, lookback: int = 10) -> bool:
    if df is None or len(df) < max(lookback, 8):
        return False
    d = df.iloc[-lookback:].copy()
    lows = d["low"].astype(float).values
    swing_lows = []
    for i in range(2, len(lows) - 2):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            swing_lows.append(lows[i])

    if len(swing_lows) >= 2:
        return float(swing_lows[-1]) >= float(swing_lows[-2])

    early_low = float(np.min(lows[: len(lows) // 2]))
    late_low = float(np.min(lows[len(lows) // 2 :]))
    return late_low >= early_low


def _range_pct(df: pd.DataFrame, lookback: int) -> float:
    if df is None or len(df) < lookback:
        return 0.0
    w = df.iloc[-lookback:]
    close = float(w.iloc[-1]["close"])
    if close <= 0:
        return 0.0
    high = float(w["high"].max())
    low = float(w["low"].min())
    return max(0.0, (high - low) / close)


def _bb_width(last_row: pd.Series) -> float:
    mid = float(last_row.get("bb_mid", 0) or 0)
    if mid <= 0:
        return 0.0
    upper = float(last_row.get("bb_upper", 0) or 0)
    lower = float(last_row.get("bb_lower", 0) or 0)
    return max(0.0, (upper - lower) / mid)


def _distance_to_resistance_pct(df_15m: pd.DataFrame, entry_price: float, lookback: int = 20) -> float:
    if df_15m is None or len(df_15m) < lookback or entry_price <= 0:
        return 0.0
    swing_high = float(df_15m.iloc[-lookback:]["high"].max())
    if swing_high <= 0:
        return 0.0
    return max(0.0, (swing_high - entry_price) / entry_price)


def _calculate_levels_from_df(df_exec: pd.DataFrame, direction: str) -> Optional[Dict[str, Any]]:
    df_closed = _closed_candle_view(df_exec, 20)
    if df_closed is None:
        return None

    d = _with_indicators(df_closed)
    last = d.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr"])
    if close <= 0 or atr <= 0:
        return None

    recent = d.iloc[-15:]
    sl_buffer_atr = float(STRATEGY_CONFIG.get("sl_buffer_atr", 0.20))

    if direction == "LONG":
        swing_low = float(recent["low"].min())
        sl_trigger = swing_low - (sl_buffer_atr * atr)
        sl_limit = sl_trigger - (0.15 * atr)
        risk = close - sl_trigger
        if risk <= 0:
            return None
        tp1 = close + (risk * float(STRATEGY_CONFIG.get("tp1_rr", 1.2)))
        tp2 = close + (risk * float(STRATEGY_CONFIG.get("tp2_rr", 2.8)))
        tp3 = close + (risk * float(STRATEGY_CONFIG.get("tp3_rr", 3.4)))
    else:
        swing_high = float(recent["high"].max())
        sl_trigger = swing_high + (sl_buffer_atr * atr)
        sl_limit = sl_trigger + (0.15 * atr)
        risk = sl_trigger - close
        if risk <= 0:
            return None
        tp1 = close - (risk * float(STRATEGY_CONFIG.get("tp1_rr", 1.2)))
        tp2 = close - (risk * float(STRATEGY_CONFIG.get("tp2_rr", 2.8)))
        tp3 = close - (risk * float(STRATEGY_CONFIG.get("tp3_rr", 3.4)))

    tp1_pct = float(STRATEGY_CONFIG.get("tp1_percent", 40))
    tp2_pct = float(STRATEGY_CONFIG.get("tp2_percent", 60))
    tp3_pct = float(STRATEGY_CONFIG.get("tp3_percent", 0))
    total_pct = max(1e-9, tp1_pct + tp2_pct + tp3_pct)
    blended_rr = (
        (float(STRATEGY_CONFIG.get("tp1_rr", 1.2)) * tp1_pct)
        + (float(STRATEGY_CONFIG.get("tp2_rr", 2.8)) * tp2_pct)
        + (float(STRATEGY_CONFIG.get("tp3_rr", 3.4)) * tp3_pct)
    ) / total_pct

    return {
        "entry": close,
        "stop_loss": sl_trigger,
        "sl_trigger": sl_trigger,
        "sl_limit": sl_limit,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr": blended_rr,
        "atr": atr,
        "tp1_percent": tp1_pct,
        "tp2_percent": tp2_pct,
        "tp3_percent": tp3_pct,
    }


def _long_trigger_5m(df_5m: pd.DataFrame) -> Tuple[bool, str, str, float]:
    df_closed = _closed_candle_view(df_5m, 40)
    if df_closed is None:
        return False, "5M veri yetersiz", "none", 0.0
    d = _with_indicators(df_closed)
    if len(d) < 40:
        return False, "5M veri yetersiz", "none", 0.0

    last = d.iloc[-1]
    prev = d.iloc[-2]

    close = float(last["close"])
    open_ = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])
    atr = float(last["atr"])
    if atr <= 0:
        return False, "5M ATR yetersiz", "none", 0.0

    rsi = float(last["rsi14"])
    macd_hist = float(last.get("macd_hist", 0) or 0)
    ema21 = float(last["ema21"])
    ema7 = float(last["ema7"])
    vol = float(last.get("volume", 0) or 0)
    vol_ma = float(last.get("vol_ma", 0) or 0)
    vol_ratio = (vol / vol_ma) if vol_ma > 0 else 0.0
    range20 = _range_pct(d, int(EXECUTION_CONFIG.get("range_5m_lookback", 20)))

    if rsi < float(EXECUTION_CONFIG.get("entry_rsi5m_min", 54.0)):
        return False, f"5M RSI düşük ({rsi:.1f})", "none", vol_ratio
    if macd_hist < 0:
        return False, f"5M MACD hist negatif ({macd_hist:.4f})", "none", vol_ratio
    if close <= ema21:
        return False, "5M fiyat EMA21 altında", "none", vol_ratio
    if range20 < float(EXECUTION_CONFIG.get("range_5m_min_pct", 0.012)):
        return False, f"5M range dar ({range20*100:.2f}%)", "none", vol_ratio

    recent_high = float(d.iloc[-21:-1]["high"].max())
    breakout = close > recent_high
    retest = low <= recent_high and close > recent_high

    body = abs(close - open_)
    lower_wick = max(0.0, min(open_, close) - low)
    rejection = close > open_ and lower_wick >= max(1e-9, body)
    local_high_break = close > float(d.iloc[-4:-1]["high"].max())
    near_ema21 = abs(close - ema21) / atr <= 0.35

    prev_sweep = float(prev["low"]) < float(d.iloc[-8:-2]["low"].min()) and float(prev["close"]) > float(d.iloc[-8:-2]["low"].min())
    sweep_trigger = prev_sweep and close > float(prev["high"])

    last3 = d.iloc[-3:]
    closes_above_ema7 = int((last3["close"] > last3["ema7"]).sum())
    min_above = int(EXECUTION_CONFIG.get("entry_min_ema7_closes", 2))
    if closes_above_ema7 < min_above:
        return False, f"EMA7 kapanış filtresi yetersiz ({closes_above_ema7}/3)", "none", vol_ratio

    if breakout and retest:
        if vol_ratio < float(EXECUTION_CONFIG.get("entry_volume_breakout_ratio", 1.2)):
            return False, f"Breakout hacim zayıf ({vol_ratio:.2f})", "none", vol_ratio
        return True, "5M Breakout+Retest", "breakout_retest", vol_ratio

    if near_ema21 and rejection and local_high_break:
        if vol_ratio < float(EXECUTION_CONFIG.get("entry_volume_retest_ratio", 1.0)):
            return False, f"Pullback hacim zayıf ({vol_ratio:.2f})", "none", vol_ratio
        return True, "5M Pullback Rejection", "pullback", vol_ratio

    if sweep_trigger:
        if vol_ratio < float(EXECUTION_CONFIG.get("entry_volume_retest_ratio", 1.0)):
            return False, f"Sweep hacim zayıf ({vol_ratio:.2f})", "none", vol_ratio
        return True, "5M Liquidity Sweep", "sweep", vol_ratio

    return False, "5M trigger yok", "none", vol_ratio


def _is_btc(symbol: str) -> bool:
    s = (symbol or "").upper()
    return "BTC" in s


def _detect_trend_5m(df_5m: pd.DataFrame) -> Tuple[str, str, float, bool]:
    """
    5m timeframe'de EMA Ribbon stack alignment ile trend tespiti.
    Returns: (direction, reason, score)
    """
    if df_5m is None or len(df_5m) < 60:
        return "NONE", "5M veri yetersiz", 0.0, False
    
    d = _with_indicators(df_5m)
    last = d.iloc[-1]
    
    ema8 = float(last["ema8"])
    ema13 = float(last["ema13"])
    ema21 = float(last["ema21"])
    ema55 = float(last["ema55"])
    rsi9 = float(last["rsi9"])
    atr = float(last["atr"])
    close = float(last["close"])
    
    if atr <= 0:
        return "NONE", "5M ATR hesaplanamadi", 0.0, False
    
    # EMA Ribbon Stack Alignment
    ribbon_dir = detect_ema_ribbon_alignment(ema8, ema13, ema21, ema55)
    
    if ribbon_dir == "LONG":
        # RSI momentum check
        if rsi9 >= 45:
            score = 2.0 if rsi9 >= 55 else 1.5
            return "LONG", f"5M EMA ribbon LONG (RSI9={rsi9:.1f})", score, False
        else:
            return "LONG", f"5M EMA ribbon LONG weak (RSI9={rsi9:.1f})", 0.8, False
    
    if ribbon_dir == "SHORT":
        if rsi9 <= 55:
            score = 2.0 if rsi9 <= 45 else 1.5
            return "SHORT", f"5M EMA ribbon SHORT (RSI9={rsi9:.1f})", score, False
        else:
            return "SHORT", f"5M EMA ribbon SHORT weak (RSI9={rsi9:.1f})", 0.8, False

    weak_gap_atr = abs(ema21 - ema55) / atr if atr > 0 else 0.0
    ema21_prev = float(d.iloc[-4]["ema21"]) if len(d) >= 4 else ema21
    ema55_prev = float(d.iloc[-4]["ema55"]) if len(d) >= 4 else ema55
    ema21_slope = (ema21 - ema21_prev) / max(abs(ema21_prev), 1e-9) * 100.0
    ema55_slope = (ema55 - ema55_prev) / max(abs(ema55_prev), 1e-9) * 100.0
    long_soft = close >= ema21 and ema21 >= ema55 and rsi9 >= 44 and weak_gap_atr >= 0.02 and ema21_slope >= -0.03 and ema55_slope >= -0.04
    short_soft = close <= ema21 and ema21 <= ema55 and rsi9 <= 56 and weak_gap_atr >= 0.02 and ema21_slope <= 0.03 and ema55_slope <= 0.04

    if long_soft:
        return "LONG", f"5M soft trend LONG (RSI9={rsi9:.1f}, gapATR={weak_gap_atr:.2f})", 0.9, True
    if short_soft:
        return "SHORT", f"5M soft trend SHORT (RSI9={rsi9:.1f}, gapATR={weak_gap_atr:.2f})", 0.9, True

    return "NONE", f"5M EMA ribbon NEUTRAL", 0.0, False


def _detect_htf_trend(df_htf: pd.DataFrame, label: str) -> Tuple[str, str, float]:
    if df_htf is None or len(df_htf) < 60:
        return "NONE", f"{label} veri yetersiz", 0.0
    
    d = _with_indicators(df_htf)
    last = d.iloc[-1]
    
    ema21 = float(last["ema21"])
    ema55 = float(last["ema55"])
    ema200 = float(last["ema200"])
    close = float(last["close"])
    rsi14 = float(last["rsi14"])
    ema21_prev = float(d.iloc[-4]["ema21"]) if len(d) >= 4 else ema21
    ema55_prev = float(d.iloc[-4]["ema55"]) if len(d) >= 4 else ema55
    ema21_slope = ema21 - ema21_prev
    ema55_slope = ema55 - ema55_prev

    if close > ema21 > ema55 and rsi14 > 50 and ema21_slope >= 0 and ema55_slope >= 0:
        score = 1.5 if close > ema200 else 1.0
        return "LONG", f"{label} trend LONG", score

    if close < ema21 < ema55 and rsi14 < 50 and ema21_slope <= 0 and ema55_slope <= 0:
        score = 1.5 if close < ema200 else 1.0
        return "SHORT", f"{label} trend SHORT", score
    
    return "NONE", f"{label} trend NEUTRAL", 0.0


def _detect_trend_15m(df_15m: pd.DataFrame) -> Tuple[str, str, float]:
    return _detect_htf_trend(df_15m, "15M")


def _detect_trend_1h(df_1h: pd.DataFrame) -> Tuple[str, str, float]:
    return _detect_htf_trend(df_1h, "1H")


def _vwap_bias(df: pd.DataFrame, direction: str) -> Tuple[bool, str, float]:
    """
    VWAP pozisyonuna göre bias filtresi.
    Fiyat VWAP üstünde ise LONG bias, altında ise SHORT bias.
    """
    if df is None or len(df) < 10:
        return True, "VWAP veri yok", 0.0
    
    d = _with_indicators(df)
    last = d.iloc[-1]
    
    close = float(last["close"])
    vwap = float(last["vwap"])
    
    if pd.isna(vwap) or vwap <= 0:
        return True, "VWAP hesaplanamadi", 0.0
    
    distance_pct = abs(close - vwap) / vwap * 100
    
    if direction == "LONG":
        if close > vwap:
            bonus = 0.8 if distance_pct < 0.5 else 0.5
            return True, f"VWAP bias OK (close > vwap)", bonus
        else:
            soft_tolerance_pct = float(EXECUTION_CONFIG.get("vwap_soft_tolerance_pct", 0.12))
            if distance_pct <= soft_tolerance_pct:
                return True, f"VWAP soft LONG (dist={distance_pct:.2f}%)", -0.15
            return False, f"VWAP bias FAIL (close < vwap)", -0.5
    
    if direction == "SHORT":
        if close < vwap:
            bonus = 0.8 if distance_pct < 0.5 else 0.5
            return True, f"VWAP bias OK (close < vwap)", bonus
        else:
            soft_tolerance_pct = float(EXECUTION_CONFIG.get("vwap_soft_tolerance_pct", 0.12))
            if distance_pct <= soft_tolerance_pct:
                return True, f"VWAP soft SHORT (dist={distance_pct:.2f}%)", -0.15
            return False, f"VWAP bias FAIL (close > vwap)", -0.5
    
    return True, "VWAP neutral", 0.0


def _cvd_confirmation(df: pd.DataFrame, direction: str) -> Tuple[bool, str, float]:
    """
    CVD (Cumulative Volume Delta) ile alici/satici baski onayı.
    CVD yukari donuyor mu (alici baski artıyor mu)?
    """
    if df is None or len(df) < 10:
        return True, "CVD veri yok", 0.0
    
    d = _with_indicators(df)
    if len(d) < 5:
        return True, "CVD yetersiz veri", 0.0
    
    cvd_current = float(d.iloc[-1]["cvd"])
    cvd_prev = float(d.iloc[-5]["cvd"])
    
    if pd.isna(cvd_current) or pd.isna(cvd_prev):
        return True, "CVD hesaplanamadi", 0.0
    
    cvd_delta = cvd_current - cvd_prev
    
    if direction == "LONG":
        if cvd_delta > 0:
            return True, f"CVD bullish (delta=+{cvd_delta:.0f})", 0.6
        else:
            return False, f"CVD bearish (delta={cvd_delta:.0f})", -0.4
    
    if direction == "SHORT":
        if cvd_delta < 0:
            return True, f"CVD bearish (delta={cvd_delta:.0f})", 0.6
        else:
            return False, f"CVD bullish (delta=+{cvd_delta:.0f})", -0.4
    
    return True, "CVD neutral", 0.0


def _volume_confirmation(df: pd.DataFrame) -> Tuple[bool, str, float]:
    if not bool(EXECUTION_CONFIG.get("volume_confirmation_enabled", True)):
        return True, "Volume filtresi pasif", 0.0

    if df is None or len(df) < 20:
        return False, "Volume veri yetersiz", -0.5

    d = _with_indicators(df)
    last = d.iloc[-1]

    volume = float(last.get("volume", 0) or 0)
    vol_ma = float(last.get("vol_ma", 0) or 0)
    min_ratio = float(EXECUTION_CONFIG.get("volume_min_ratio", 0.85))

    if vol_ma <= 0:
        return False, "Volume MA hesaplanamadi", -0.5

    ratio = volume / vol_ma
    if ratio >= min_ratio:
        score = 0.7 if ratio >= max(1.10, min_ratio + 0.15) else 0.3
        return True, f"Volume OK (ratio={ratio:.2f})", score

    soft_floor = float(EXECUTION_CONFIG.get("volume_soft_floor_ratio", max(0.50, min_ratio - 0.16)))
    if ratio >= soft_floor:
        return True, f"Volume soft OK (ratio={ratio:.2f}<{min_ratio:.2f})", -0.12

    return False, f"Volume fail (ratio={ratio:.2f}<{min_ratio:.2f})", -0.6


def _adx_confirmation(df: pd.DataFrame, direction: str) -> Tuple[bool, str, float]:
    if not bool(EXECUTION_CONFIG.get("adx_filter_enabled", True)):
        return True, "ADX filtresi pasif", 0.0

    if df is None or len(df) < 20:
        return False, "ADX veri yetersiz", -0.5

    d = _with_indicators(df)
    last = d.iloc[-1]

    adx = float(last.get("adx", 0) or 0)
    adx_pos = float(last.get("adx_pos", 0) or 0)
    adx_neg = float(last.get("adx_neg", 0) or 0)
    min_adx = float(EXECUTION_CONFIG.get("adx_min_value", 20))
    soft_floor = float(EXECUTION_CONFIG.get("adx_soft_floor", max(10.0, min_adx - 4.0)))

    if adx < min_adx:
        if adx >= soft_floor:
            return True, f"ADX soft OK ({adx:.1f}<{min_adx:.1f})", -0.2
        return False, f"ADX fail ({adx:.1f}<{min_adx:.1f})", -0.6

    if direction == "LONG" and adx_pos < adx_neg:
        if abs(adx_pos - adx_neg) <= 2.0:
            return True, f"ADX yön soft LONG (+DI={adx_pos:.1f}, -DI={adx_neg:.1f})", -0.15
        return False, f"ADX yön fail (+DI={adx_pos:.1f} < -DI={adx_neg:.1f})", -0.5

    if direction == "SHORT" and adx_neg < adx_pos:
        if abs(adx_neg - adx_pos) <= 2.0:
            return True, f"ADX yön soft SHORT (-DI={adx_neg:.1f}, +DI={adx_pos:.1f})", -0.15
        return False, f"ADX yön fail (-DI={adx_neg:.1f} < +DI={adx_pos:.1f})", -0.5

    return True, f"ADX OK (adx={adx:.1f})", 0.6


def _market_structure_confirmation(df: pd.DataFrame, direction: str) -> Tuple[bool, str, float]:
    if not bool(EXECUTION_CONFIG.get("market_structure_enabled", True)):
        return True, "Structure filtresi pasif", 0.0

    lookback = max(3, int(EXECUTION_CONFIG.get("market_structure_lookback", 5)))
    min_len = (lookback * 2) + 5
    if df is None or len(df) < min_len:
        return False, "Structure veri yetersiz", -0.5

    d = _with_indicators(df)
    prev = d.iloc[-(lookback * 2):-lookback]
    recent = d.iloc[-lookback:]
    last = d.iloc[-1]

    prev_high = float(prev["high"].max())
    prev_low = float(prev["low"].min())
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    close = float(last["close"])
    ema21 = float(last["ema21"])
    atr = max(float(last.get("atr", 0) or 0), 1e-9)
    tolerance = 0.15 * atr

    if direction == "LONG":
        higher_low = recent_low >= (prev_low - tolerance)
        improving_high = recent_high >= (prev_high - tolerance)
        hold_ema = close >= (ema21 - (0.05 * atr))
        passed = int(higher_low) + int(hold_ema) + int(improving_high)
        ok = passed == 3
        soft_ok = passed >= 2
        reason = (
            f"Structure LONG {'OK' if ok else ('SOFT' if soft_ok else 'FAIL')} "
            f"(hl={int(higher_low)}, hh={int(improving_high)}, hold={int(hold_ema)})"
        )
        if ok:
            return True, reason, 0.7
        if soft_ok:
            return True, reason, -0.1
        return False, reason, -0.7

    lower_high = recent_high <= (prev_high + tolerance)
    improving_low = recent_low <= (prev_low + tolerance)
    hold_ema = close <= (ema21 + (0.05 * atr))
    passed = int(lower_high) + int(hold_ema) + int(improving_low)
    ok = passed == 3
    soft_ok = passed >= 2
    reason = (
        f"Structure SHORT {'OK' if ok else ('SOFT' if soft_ok else 'FAIL')} "
        f"(lh={int(lower_high)}, ll={int(improving_low)}, hold={int(hold_ema)})"
    )
    if ok:
        return True, reason, 0.7
    if soft_ok:
        return True, reason, -0.1
    return False, reason, -0.7




def _entry_trigger_1m(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    direction: str,
) -> Tuple[bool, str, float, Optional[Dict[str, Any]]]:
    """
    1m timeframe'de entry trigger tespiti.
    
    Koşullar:
    1. EMA ribbon pullback - fiyat EMA21'e çekiliyor
    2. Engulfing candle + hacim artışı
    3. RSI9 optimal aralıkta
    4. ATR-based body ratio
    
    Returns: (triggered, reason, entry_score, levels)
    """
    df_1m_closed = _closed_candle_view(df_1m, 30)
    df_5m_closed = _closed_candle_view(df_5m, 30) if df_5m is not None else None

    if df_1m_closed is None:
        return False, "1M veri yetersiz", 0.0, None
    
    d1m = _with_indicators(df_1m_closed)
    d5m = _with_indicators(df_5m_closed) if df_5m_closed is not None else None
    
    last = d1m.iloc[-1]
    prev = d1m.iloc[-2] if len(d1m) >= 2 else last
    
    close = float(last["close"])
    open_price = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])
    ema8 = float(last["ema8"])
    ema21 = float(last["ema21"])
    ema55 = float(last["ema55"])
    rsi9 = float(last["rsi9"])
    atr = float(last["atr"])
    volume = float(last.get("volume", 0) or 0)
    vol_ma = float(last.get("vol_ma", 1) or 1)
    
    prev_close = float(prev["close"])
    prev_open = float(prev["open"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    
    if atr <= 0:
        return False, "1M ATR hesaplanamadi", 0.0, None
    
    # Body ratio check
    body = abs(close - open_price)
    candle_range = max(1e-9, high - low)
    body_ratio = body / candle_range
    
    # Volume spike check
    volume_ratio = volume / vol_ma if vol_ma > 0 else 0
    volume_spike_ratio = float(EXECUTION_CONFIG.get("entry_volume_spike_ratio", 1.15))
    has_volume_spike = volume_ratio >= volume_spike_ratio
    
    # EMA distance for pullback detection
    ema21_distance_atr = abs(close - ema21) / atr
    
    entry_score = 0.0
    reasons = []
    
    pullback_band_atr = float(EXECUTION_CONFIG.get("trigger_pullback_band_atr", 0.25))
    allow_continuation = bool(EXECUTION_CONFIG.get("allow_continuation_entry", True))
    continuation_min_body_ratio = float(EXECUTION_CONFIG.get("continuation_min_body_ratio", 0.58))
    continuation_min_close_location = float(EXECUTION_CONFIG.get("continuation_min_close_location", 0.60))
    continuation_breakout_hold_atr = float(EXECUTION_CONFIG.get("continuation_breakout_hold_atr", 0.10))
    continuation_require_volume_spike = bool(EXECUTION_CONFIG.get("continuation_require_volume_spike", False))
    continuation_lookback = max(4, int(EXECUTION_CONFIG.get("continuation_impulse_lookback_candles", 16)))
    continuation_require_ema_alignment = bool(EXECUTION_CONFIG.get("continuation_require_ema_alignment", True))
    continuation_require_ema_slope = bool(EXECUTION_CONFIG.get("continuation_require_ema_slope", True))
    continuation_ema_slope_lookback = max(2, int(EXECUTION_CONFIG.get("continuation_ema_slope_lookback", 3)))
    long_rsi_min = float(EXECUTION_CONFIG.get("entry_rsi_long_min", 45))
    long_rsi_max = float(EXECUTION_CONFIG.get("entry_rsi_long_max", 68))
    short_rsi_min = float(EXECUTION_CONFIG.get("entry_rsi_short_min", 38))
    short_rsi_max = float(EXECUTION_CONFIG.get("entry_rsi_short_max", 55))
    strong_body_ratio = max(0.45, float(STRATEGY_CONFIG.get("strong_body_ratio", 0.55)))
    min_trigger_body_ratio = float(EXECUTION_CONFIG.get("trigger_min_body_ratio", 0.35))
    close_location = (close - low) / candle_range if direction == "LONG" else (high - close) / candle_range
    min_close_location = float(EXECUTION_CONFIG.get("trigger_min_close_location", 0.45))

    if direction == "LONG":
        # Engulfing pattern
        is_engulfing = (
            close > open_price and
            close > prev_high and
            open_price < prev_low and
            body_ratio >= 0.5
        )
        
        # EMA pullback entry - fiyat EMA21'e yakın ve yukarı dönüyor
        near_ema21 = ema21_distance_atr <= pullback_band_atr
        above_ema21 = close > ema21
        ema_alignment = ema8 > ema21 > ema55
        reclaimed_ema21 = low <= ema21 and close > ema21

        # RSI9 optimal range for LONG entry
        if rsi9 > long_rsi_max:
            return False, f"LONG entry blocked: RSI9 çok yüksek ({rsi9:.1f}>{long_rsi_max:.1f})", 0.0, None
        rsi_ok = long_rsi_min <= rsi9 <= long_rsi_max

        # Bullish candle
        bullish = close > open_price
        strong_momentum_candle = bullish and body_ratio >= strong_body_ratio and close_location >= min_close_location

        history = d1m.iloc[-(continuation_lookback + 1):-1] if len(d1m) > continuation_lookback else d1m.iloc[:-1]
        impulse_high = float(history["high"].max()) if len(history) > 0 else prev_high
        ema21_prev_idx = -1 - continuation_ema_slope_lookback
        ema8_prev = float(d1m.iloc[ema21_prev_idx]["ema8"]) if len(d1m) > continuation_ema_slope_lookback else ema8
        ema21_prev_slope = float(d1m.iloc[ema21_prev_idx]["ema21"]) if len(d1m) > continuation_ema_slope_lookback else ema21
        ema8_slope_ok = ema8 >= ema8_prev
        ema21_slope_ok = ema21 >= ema21_prev_slope
        ema_slope_ok = ema8_slope_ok and ema21_slope_ok
        continuation_breakout_ok = high >= impulse_high and close >= (impulse_high - (continuation_breakout_hold_atr * atr))
        recent_lows = d1m["low"].iloc[-6:]
        prior_lows = d1m["low"].iloc[-12:-6]
        continuation_structure_ok = True
        if len(recent_lows) >= 3 and len(prior_lows) >= 3:
            continuation_structure_ok = float(recent_lows.min()) >= (float(prior_lows.min()) - (0.12 * atr))
        continuation_momentum_ok = strong_momentum_candle or has_volume_spike
        if (not continuation_require_volume_spike) and body_ratio >= continuation_min_body_ratio and close_location >= continuation_min_close_location:
            continuation_momentum_ok = True
        continuation_quality_ok = (
            allow_continuation
            and above_ema21
            and bullish
            and rsi_ok
            and body_ratio >= continuation_min_body_ratio
            and close_location >= continuation_min_close_location
            and continuation_breakout_ok
            and continuation_momentum_ok
            and continuation_structure_ok
            and ((not continuation_require_ema_alignment) or ema_alignment)
            and ((not continuation_require_ema_slope) or ema_slope_ok)
        )

        if not reclaimed_ema21 and not near_ema21 and not (is_engulfing and has_volume_spike) and not continuation_quality_ok:
            return False, "LONG entry pending: retest/reclaim yok", 0.0, None

        reclaim_or_pullback = reclaimed_ema21 or (near_ema21 and above_ema21)

        # Scoring
        if is_engulfing and has_volume_spike:
            entry_score += 2.8
            reasons.append("engulfing+volume")
        elif strong_momentum_candle and has_volume_spike:
            entry_score += 1.9
            reasons.append("strong_bullish+volume")
        elif bullish and body_ratio >= 0.4:
            entry_score += 1.2
            reasons.append("bullish_candle")

        if reclaimed_ema21:
            entry_score += 2.2
            reasons.append("ema21_reclaim")
        elif near_ema21 and above_ema21:
            entry_score += 1.8
            reasons.append("ema21_pullback")
        elif above_ema21:
            entry_score += 0.4
            reasons.append("above_ema21")

        if ema_alignment:
            entry_score += 1.0
            reasons.append("ema_stack")

        if rsi_ok:
            entry_score += 0.7
            reasons.append(f"rsi9={rsi9:.0f}")

        if has_volume_spike:
            entry_score += 0.4
            reasons.append("volume_spike")

        if continuation_quality_ok:
            entry_score += 1.0
            reasons.append("structure_continuation")

        # Minimum score requirement
        min_score = float(EXECUTION_CONFIG.get("min_entry_score", 4.0))
        continuation_min_score = float(
            EXECUTION_CONFIG.get("continuation_min_entry_score", max(3.0, min_score - 0.7))
        )
        trigger_quality_ok = (
            bullish
            and rsi_ok
            and body_ratio >= min_trigger_body_ratio
            and close_location >= min_close_location
            and (has_volume_spike or is_engulfing or strong_momentum_candle)
            and (
                (reclaim_or_pullback and (ema_alignment or is_engulfing))
                or continuation_quality_ok
            )
        )
        continuation_triggered = continuation_quality_ok and entry_score >= continuation_min_score
        triggered = continuation_triggered or (entry_score >= min_score and trigger_quality_ok)

        if triggered:
            # Calculate SL/TP levels
            levels = _calculate_levels_1m(d1m, d5m, direction)
            if continuation_triggered:
                reason = f"LONG Continuation+Triggered (score={entry_score:.1f}>={continuation_min_score:.1f}): {', '.join(reasons)}"
            else:
                reason = f"LONG Retest+Triggered (score={entry_score:.1f}>={min_score:.1f}): {', '.join(reasons)}"
            return True, reason, entry_score, levels
        else:
            gate_reason = []
            if not reclaim_or_pullback:
                gate_reason.append("reclaim/pullback")
            if not (ema_alignment or is_engulfing):
                gate_reason.append("ema_stack")
            if not rsi_ok:
                gate_reason.append("rsi")
            if body_ratio < min_trigger_body_ratio:
                gate_reason.append("body")
            if close_location < min_close_location:
                gate_reason.append("close_location")
            if not (has_volume_spike or is_engulfing or strong_momentum_candle):
                gate_reason.append("momentum")
            if not reclaim_or_pullback and not continuation_quality_ok:
                gate_reason.append("continuation")
            gate_text = ", ".join(gate_reason) if gate_reason else "score"
            reason = (
                f"LONG entry pending (score={entry_score:.1f}<{min_score:.1f}, "
                f"cont_score_need={continuation_min_score:.1f}, gate={gate_text}): "
                f"{', '.join(reasons) if reasons else 'no_signals'}"
            )
            return False, reason, entry_score, None

    elif direction == "SHORT":
        # Bearish engulfing
        is_engulfing = (
            close < open_price and
            close < prev_low and
            open_price > prev_high and
            body_ratio >= 0.5
        )

        # EMA pullback entry - fiyat EMA21'e yakın ve aşağı dönüyor
        near_ema21 = ema21_distance_atr <= pullback_band_atr
        below_ema21 = close < ema21
        ema_alignment = ema8 < ema21 < ema55
        reclaimed_ema21 = high >= ema21 and close < ema21

        # RSI9 optimal range for SHORT entry
        if rsi9 < short_rsi_min:
            return False, f"SHORT entry blocked: RSI9 çok düşük ({rsi9:.1f}<{short_rsi_min:.1f})", 0.0, None
        rsi_ok = short_rsi_min <= rsi9 <= short_rsi_max

        # Bearish candle
        bearish = close < open_price
        strong_momentum_candle = bearish and body_ratio >= strong_body_ratio and close_location >= min_close_location

        history = d1m.iloc[-(continuation_lookback + 1):-1] if len(d1m) > continuation_lookback else d1m.iloc[:-1]
        impulse_low = float(history["low"].min()) if len(history) > 0 else prev_low
        ema21_prev_idx = -1 - continuation_ema_slope_lookback
        ema8_prev = float(d1m.iloc[ema21_prev_idx]["ema8"]) if len(d1m) > continuation_ema_slope_lookback else ema8
        ema21_prev_slope = float(d1m.iloc[ema21_prev_idx]["ema21"]) if len(d1m) > continuation_ema_slope_lookback else ema21
        ema8_slope_ok = ema8 <= ema8_prev
        ema21_slope_ok = ema21 <= ema21_prev_slope
        ema_slope_ok = ema8_slope_ok and ema21_slope_ok
        continuation_breakout_ok = low <= impulse_low and close <= (impulse_low + (continuation_breakout_hold_atr * atr))
        recent_highs = d1m["high"].iloc[-6:]
        prior_highs = d1m["high"].iloc[-12:-6]
        continuation_structure_ok = True
        if len(recent_highs) >= 3 and len(prior_highs) >= 3:
            continuation_structure_ok = float(recent_highs.max()) <= (float(prior_highs.max()) + (0.12 * atr))
        continuation_momentum_ok = strong_momentum_candle or has_volume_spike
        if (not continuation_require_volume_spike) and body_ratio >= continuation_min_body_ratio and close_location >= continuation_min_close_location:
            continuation_momentum_ok = True
        continuation_quality_ok = (
            allow_continuation
            and below_ema21
            and bearish
            and rsi_ok
            and body_ratio >= continuation_min_body_ratio
            and close_location >= continuation_min_close_location
            and continuation_breakout_ok
            and continuation_momentum_ok
            and continuation_structure_ok
            and ((not continuation_require_ema_alignment) or ema_alignment)
            and ((not continuation_require_ema_slope) or ema_slope_ok)
        )

        if not reclaimed_ema21 and not near_ema21 and not (is_engulfing and has_volume_spike) and not continuation_quality_ok:
            return False, "SHORT entry pending: retest/reclaim yok", 0.0, None

        reclaim_or_pullback = reclaimed_ema21 or (near_ema21 and below_ema21)

        # Scoring
        if is_engulfing and has_volume_spike:
            entry_score += 2.8
            reasons.append("engulfing+volume")
        elif strong_momentum_candle and has_volume_spike:
            entry_score += 1.9
            reasons.append("strong_bearish+volume")
        elif bearish and body_ratio >= 0.4:
            entry_score += 1.2
            reasons.append("bearish_candle")

        if reclaimed_ema21:
            entry_score += 2.2
            reasons.append("ema21_reclaim")
        elif near_ema21 and below_ema21:
            entry_score += 2.0
            reasons.append("ema21_pullback")
        elif below_ema21:
            entry_score += 0.4
            reasons.append("below_ema21")

        if ema_alignment:
            entry_score += 1.0
            reasons.append("ema_stack")

        if rsi_ok:
            entry_score += 0.7
            reasons.append(f"rsi9={rsi9:.0f}")

        if has_volume_spike:
            entry_score += 0.4
            reasons.append("volume_spike")

        if continuation_quality_ok:
            entry_score += 1.0
            reasons.append("structure_continuation")

        # Minimum score requirement
        min_score = float(EXECUTION_CONFIG.get("min_entry_score", 4.0))
        continuation_min_score = float(
            EXECUTION_CONFIG.get("continuation_min_entry_score", max(3.0, min_score - 0.7))
        )
        trigger_quality_ok = (
            bearish
            and rsi_ok
            and body_ratio >= min_trigger_body_ratio
            and close_location >= min_close_location
            and (has_volume_spike or is_engulfing or strong_momentum_candle)
            and (
                (reclaim_or_pullback and (ema_alignment or is_engulfing))
                or continuation_quality_ok
            )
        )
        continuation_triggered = continuation_quality_ok and entry_score >= continuation_min_score
        triggered = continuation_triggered or (entry_score >= min_score and trigger_quality_ok)

        if triggered:
            # Calculate SL/TP levels
            levels = _calculate_levels_1m(d1m, d5m, direction)
            if continuation_triggered:
                reason = f"SHORT Continuation+Triggered (score={entry_score:.1f}>={continuation_min_score:.1f}): {', '.join(reasons)}"
            else:
                reason = f"SHORT Retest+Triggered (score={entry_score:.1f}>={min_score:.1f}): {', '.join(reasons)}"
            return True, reason, entry_score, levels
        else:
            gate_reason = []
            if not reclaim_or_pullback:
                gate_reason.append("reclaim/pullback")
            if not (ema_alignment or is_engulfing):
                gate_reason.append("ema_stack")
            if not rsi_ok:
                gate_reason.append("rsi")
            if body_ratio < min_trigger_body_ratio:
                gate_reason.append("body")
            if close_location < min_close_location:
                gate_reason.append("close_location")
            if not (has_volume_spike or is_engulfing or strong_momentum_candle):
                gate_reason.append("momentum")
            if not reclaim_or_pullback and not continuation_quality_ok:
                gate_reason.append("continuation")
            gate_text = ", ".join(gate_reason) if gate_reason else "score"
            reason = (
                f"SHORT entry pending (score={entry_score:.1f}<{min_score:.1f}, "
                f"cont_score_need={continuation_min_score:.1f}, gate={gate_text}): "
                f"{', '.join(reasons) if reasons else 'no_signals'}"
            )
            return False, reason, entry_score, None
    
    return False, "Invalid direction", 0.0, None


def _calculate_levels_1m(
    df_1m: pd.DataFrame,
    df_5m: Optional[pd.DataFrame],
    direction: str,
) -> Optional[Dict[str, Any]]:
    """
    1m timeframe bazında SL/TP seviyelerini hesaplar.
    
    SL: Son 10-20 bar swing low/high + ATR buffer
    TP: RR bazlı (1.5R, 2.1R, 2.8R)
    """
    df_1m_closed = _closed_candle_view(df_1m, 20)
    if df_1m_closed is None:
        return None
    
    d = _with_indicators(df_1m_closed)
    last = d.iloc[-1]
    
    close = float(last["close"])
    atr = float(last["atr"])
    
    if atr <= 0 or close <= 0:
        return None
    
    # Swing lookback for SL
    lookback = 15
    recent = d.iloc[-lookback:]
    
    sl_buffer_atr = float(STRATEGY_CONFIG.get("sl_buffer_atr", 0.20))
    
    if direction == "LONG":
        swing_low = float(recent["low"].min())
        sl_trigger = swing_low - (sl_buffer_atr * atr)
        sl_limit = sl_trigger - (0.15 * atr)
        risk = close - sl_trigger
        if risk <= 0:
            sl_trigger = close - max(0.85 * atr, 1e-9)
            sl_limit = sl_trigger - (0.15 * atr)
            risk = close - sl_trigger
    else:
        swing_high = float(recent["high"].max())
        sl_trigger = swing_high + (sl_buffer_atr * atr)
        sl_limit = sl_trigger + (0.15 * atr)
        risk = sl_trigger - close
        if risk <= 0:
            sl_trigger = close + max(0.85 * atr, 1e-9)
            sl_limit = sl_trigger + (0.15 * atr)
            risk = sl_trigger - close
    
    if risk <= 0:
        return None
    
    # TP ratios - kullanıcının isteği: 1.5R, 2.1R, 2.8R
    tp1_rr = float(STRATEGY_CONFIG.get("tp1_rr", 1.5))
    tp2_rr = float(STRATEGY_CONFIG.get("tp2_rr", 2.1))
    tp3_rr = float(STRATEGY_CONFIG.get("tp3_rr", 2.8))
    
    if direction == "LONG":
        tp1 = close + (risk * tp1_rr)
        tp2 = close + (risk * tp2_rr)
        tp3 = close + (risk * tp3_rr)
    else:
        tp1 = close - (risk * tp1_rr)
        tp2 = close - (risk * tp2_rr)
        tp3 = close - (risk * tp3_rr)
    
    # TP percentages
    tp1_pct = float(STRATEGY_CONFIG.get("tp1_percent", 40))
    tp2_pct = float(STRATEGY_CONFIG.get("tp2_percent", 35))
    tp3_pct = float(STRATEGY_CONFIG.get("tp3_percent", 25))
    
    # Blended RR
    total_pct = tp1_pct + tp2_pct + tp3_pct
    blended_rr = (tp1_rr * tp1_pct + tp2_rr * tp2_pct + tp3_rr * tp3_pct) / total_pct
    
    return {
        "entry": close,
        "stop_loss": sl_trigger,
        "sl_trigger": sl_trigger,
        "sl_limit": sl_limit,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr": blended_rr,
        "atr": atr,
        "tp1_percent": tp1_pct,
        "tp2_percent": tp2_pct,
        "tp3_percent": tp3_pct,
    }


def analyze(
    trigger_df: pd.DataFrame,
    current_time: tuple,
    bias_df: Optional[pd.DataFrame] = None,
    mtf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    mtf_enabled: bool = True,
    min_mtf_confirmations: int = 2,
    symbol: str = "",
) -> Optional[Dict[str, Any]]:
    del bias_df, mtf_enabled, min_mtf_confirmations

    hour, _, _, _ = current_time
    if not is_time_allowed(hour):
        return {"signal": None, "reason": "Izin verilen saatler dışında"}

    df_5m = _closed_candle_view((mtf_dfs or {}).get("5") if mtf_dfs else trigger_df, 80)
    if df_5m is None:
        df_5m = _closed_candle_view(trigger_df, 80)
    df_15m = _closed_candle_view((mtf_dfs or {}).get("15") if mtf_dfs else None, 80)
    df_1h = _closed_candle_view((mtf_dfs or {}).get("60") if mtf_dfs else None, 120)

    if df_5m is None or len(df_5m) < 80:
        return {"signal": None, "reason": "5M veri yetersiz"}
    if df_15m is None or len(df_15m) < 80:
        return {"signal": None, "reason": "15M veri yetersiz"}
    if df_1h is None or len(df_1h) < 120:
        return {"signal": None, "reason": "1H veri yetersiz"}

    direction = "LONG"
    if not bool(EXECUTION_CONFIG.get("long_only_mode", True)):
        direction = "LONG"

    reasons: list[str] = []
    d1h = _with_indicators(df_1h)
    d15 = _with_indicators(df_15m)
    d5 = _with_indicators(df_5m)

    # 1H direction + pump filters
    last1h = d1h.iloc[-1]
    ema50_1h = float(last1h["ema55"])
    ema200_1h = float(last1h["ema200"])
    ema21_1h = float(last1h["ema21"])
    close_1h = float(last1h["close"])
    sep = ((ema50_1h - ema200_1h) / ema200_1h) if ema200_1h > 0 else 0.0
    min_sep = float(EXECUTION_CONFIG.get("long_1h_ema_separation_min", 0.003))

    if not (ema50_1h > ema200_1h and close_1h > ema21_1h and sep > min_sep):
        return {
            "signal": None,
            "reason": (
                f"1H yön filtresi fail (ema50={ema50_1h:.2f}, ema200={ema200_1h:.2f}, "
                f"close={close_1h:.2f}, sep={sep*100:.2f}%)"
            ),
        }
    reasons.append(f"1H trend OK (sep={sep*100:.2f}%)")

    low12 = float(d1h.iloc[-12:]["low"].min())
    low24 = float(d1h.iloc[-24:]["low"].min())
    pump12 = ((close_1h - low12) / low12) if low12 > 0 else 0.0
    pump24 = ((close_1h - low24) / low24) if low24 > 0 else 0.0
    th12 = float(EXECUTION_CONFIG.get("pump_lookback_12_threshold", 0.12))
    th24 = float(EXECUTION_CONFIG.get("pump_lookback_24_threshold", 0.18))
    if pump12 > th12 or pump24 > th24:
        return {
            "signal": None,
            "reason": f"Pump filtresi fail (12h={pump12*100:.1f}%, 24h={pump24*100:.1f}%)",
        }
    reasons.append(f"Pump OK (12h={pump12*100:.1f}%, 24h={pump24*100:.1f}%)")

    # 15M setup filters
    last15 = d15.iloc[-1]
    close15 = float(last15["close"])
    ema21_15 = float(last15["ema21"])
    ema50_15 = float(last15["ema55"])
    rsi15 = float(last15["rsi14"])
    macd15 = float(last15.get("macd_hist", 0) or 0)
    hl_ok = _higher_low_intact(d15, int(EXECUTION_CONFIG.get("setup_hl_lookback", 10)))

    if not (close15 > ema21_15 and ema21_15 > ema50_15 and hl_ok and rsi15 > float(EXECUTION_CONFIG.get("setup_rsi15_min", 52.0)) and macd15 >= 0):
        return {
            "signal": None,
            "reason": (
                "15M setup fail "
                f"(close>ema21={int(close15 > ema21_15)}, ema21>ema50={int(ema21_15 > ema50_15)}, "
                f"hl={int(hl_ok)}, rsi={rsi15:.1f}, macd={macd15:.4f})"
            ),
        }
    reasons.append("15M setup OK")

    # 15M + 5M range filters
    range15 = _range_pct(d15, int(EXECUTION_CONFIG.get("range_15m_lookback", 12)))
    atr15_pct = (float(last15["atr"]) / close15) if close15 > 0 else 0.0
    bb15 = _bb_width(last15)
    if range15 < float(EXECUTION_CONFIG.get("range_15m_min_pct", 0.018)):
        return {"signal": None, "reason": f"15M range dar ({range15*100:.2f}%)"}
    if atr15_pct < float(EXECUTION_CONFIG.get("atr_15m_min_pct", 0.006)):
        return {"signal": None, "reason": f"15M ATR düşük ({atr15_pct*100:.2f}%)"}
    if bb15 < float(EXECUTION_CONFIG.get("bb_width_min", 0.015)):
        return {"signal": None, "reason": f"15M BB width düşük ({bb15*100:.2f}%)"}

    last5 = d5.iloc[-1]
    range5 = _range_pct(d5, int(EXECUTION_CONFIG.get("range_5m_lookback", 20)))
    bb5 = _bb_width(last5)
    if range5 < float(EXECUTION_CONFIG.get("range_5m_min_pct", 0.012)):
        return {"signal": None, "reason": f"5M range dar ({range5*100:.2f}%)"}
    if bb5 < float(EXECUTION_CONFIG.get("bb_width_min", 0.015)):
        return {"signal": None, "reason": f"5M BB width düşük ({bb5*100:.2f}%)"}
    reasons.append(f"Range OK (15m={range15*100:.2f}%, 5m={range5*100:.2f}%)")

    # Resistance distance filter
    entry_ref = float(last5["close"])
    resistance_dist = _distance_to_resistance_pct(d15, entry_ref, int(EXECUTION_CONFIG.get("resistance_lookback", 20)))
    min_res_dist = float(EXECUTION_CONFIG.get("resistance_min_distance_pct", 0.008))
    if resistance_dist < min_res_dist:
        return {
            "signal": None,
            "reason": f"Dirence çok yakın (mesafe={resistance_dist*100:.2f}% < {min_res_dist*100:.2f}%)",
        }
    reasons.append(f"Resistance distance OK ({resistance_dist*100:.2f}%)")

    # 5M trigger
    triggered, trigger_reason, trigger_type, volume_ratio = _long_trigger_5m(d5)
    levels = _calculate_levels_from_df(d5, direction)
    if not levels:
        return {"signal": None, "reason": "SL/TP hesaplanamadi"}

    rr = float(levels.get("rr", 0) or 0)
    min_rr = float(STRATEGY_CONFIG.get("min_rr", 1.2))
    if rr < min_rr:
        return {"signal": None, "reason": f"RR düşük ({rr:.2f}<{min_rr:.2f})"}

    total_score = 0.0
    total_score += min(2.0, sep * 300)
    total_score += min(1.5, max(0.0, resistance_dist * 100))
    total_score += min(1.5, max(0.0, range15 * 100))
    total_score += min(1.5, max(0.0, range5 * 100))
    total_score += 1.2 if triggered else 0.4
    total_score += 0.5 if volume_ratio >= float(EXECUTION_CONFIG.get("entry_volume_breakout_ratio", 1.2)) else 0.0
    min_total_score = float(STRATEGY_CONFIG.get("min_total_score", 3.8))

    if total_score < min_total_score:
        return {
            "signal": None,
            "reason": f"Yetersiz puan ({total_score:.1f}<{min_total_score:.1f})",
            "total_score": total_score,
        }

    breakout_level = float(d5.iloc[-21:-1]["high"].max()) if len(d5) >= 25 else float(last5["high"])

    if not triggered:
        return {
            "signal": direction,
            "reason": f"5M entry trigger bekleniyor (QUEUED) | {trigger_reason}",
            "setup_state": "QUEUED",
            "entry_state": "Pending",
            "trigger_reason": trigger_reason,
            "trigger_type": trigger_type,
            "breakout_level": breakout_level,
            "quality_score": total_score,
            "total_score": total_score,
            "risk_multiplier": 1.0,
            "trend_state": direction,
            "mtf_confirmed": True,
            "trend_strength": "medium",
            "spread_bps": 0.0,
            "level_source": "5m_exec",
            **levels,
        }

    final_reason = f"{' | '.join(reasons)} | {trigger_reason} | score={total_score:.1f}"
    return {
        "signal": direction,
        "reason": final_reason,
        "setup_state": "TRIGGERED",
        "entry_state": "TRIGGERED",
        "trigger_reason": trigger_reason,
        "trigger_type": trigger_type,
        "breakout_level": breakout_level,
        "quality_score": total_score,
        "total_score": total_score,
        "risk_multiplier": 1.0,
        "trend_state": direction,
        "mtf_confirmed": True,
        "trend_strength": "strong" if total_score >= 7.0 else "medium",
        "spread_bps": 0.0,
        "level_source": "5m_exec",
        **levels,
    }


def get_execution_trigger(
    df: pd.DataFrame,
    signal: str,
    breakout_level: Optional[float] = None,
    mtf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    current_time: Optional[tuple] = None,
    symbol: str = "",
) -> Tuple[bool, str]:
    del breakout_level

    direction = (signal or "").upper()
    if direction not in ("LONG", "SHORT"):
        return False, "Yön geçersiz"

    if df is None or len(df) < 10:
        return False, "Veri yetersiz"

    if mtf_dfs:
        if current_time is None:
            ts = pd.Timestamp.utcnow()
            current_time = (ts.hour, ts.minute, ts.day, ts.weekday())

        result = analyze(
            df,
            current_time,
            mtf_dfs=mtf_dfs,
            symbol=symbol,
        )

        if not result:
            return False, "Strateji re-check sonucu yok"

        if result.get("signal") != direction:
            return False, str(result.get("reason", "Yön değişti veya setup bozuldu"))

        if str(result.get("setup_state", "")).upper() == "TRIGGERED":
            return True, str(result.get("trigger_reason", result.get("reason", "TRIGGERED")))

        return False, str(result.get("reason", result.get("trigger_reason", "Trigger bekleniyor")))

    if direction != "LONG":
        return False, "Sadece LONG aktif"

    triggered, reason, _, _ = _long_trigger_5m(df)
    return triggered, reason
