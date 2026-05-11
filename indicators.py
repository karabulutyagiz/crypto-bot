import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import numpy as np
from typing import Tuple


def calculate_ema(close: pd.Series, length: int = 50) -> pd.Series:
    ema = EMAIndicator(close=close, window=length)
    return ema.ema_indicator()


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    rsi = RSIIndicator(close=close, window=length)
    return rsi.rsi()


def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd = MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    atr = AverageTrueRange(high=high, low=low, close=close, window=length)
    return atr.average_true_range()


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    adx_indicator = ADXIndicator(high=high, low=low, close=close, window=length)
    return (
        adx_indicator.adx(),
        adx_indicator.adx_pos(),
        adx_indicator.adx_neg(),
    )


def calculate_volume_ma(volume: pd.Series, length: int = 20) -> pd.Series:
    return volume.rolling(window=length).mean()


def detect_volume_spike(volume: pd.Series, volume_ma: pd.Series, multiplier: float = 1.5) -> pd.Series:
    return volume > (volume_ma * multiplier)


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    VWAP (Volume Weighted Average Price) hesaplar.
    Rolling 50-bar pencere kullanir (intraday reset yerine).
    """
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    window = 50
    cum_tp_vol = tp_vol.rolling(window=window, min_periods=1).sum()
    cum_vol = volume.rolling(window=window, min_periods=1).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.fillna(typical_price)


def calculate_cvd(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Cumulative Volume Delta (CVD) - alici/satici baski gostergesi.
    Fiyat yukari kapanirsa hacim pozitif, asagi kapanirsa negatif sayilir.
    """
    delta = pd.Series(0.0, index=close.index)
    prev_close = close.shift(1)
    bull_mask = close >= prev_close
    bear_mask = close < prev_close
    delta[bull_mask] = volume[bull_mask]
    delta[bear_mask] = -volume[bear_mask]
    return delta.cumsum()


def calculate_ema_ribbon(close: pd.Series) -> dict:
    """
    EMA Ribbon: 8, 13, 21, 55 periyot - scalp momentum icin optimize edilmis.
    """
    return {
        "ema8": calculate_ema(close, 8),
        "ema13": calculate_ema(close, 13),
        "ema21": calculate_ema(close, 21),
        "ema55": calculate_ema(close, 55),
    }


def detect_ema_ribbon_alignment(ema8: float, ema13: float, ema21: float, ema55: float) -> str:
    """
    EMA ribbon stack yonunu tespit eder.
    Returns: 'LONG', 'SHORT', or 'NEUTRAL'
    """
    if ema8 > ema13 > ema21 > ema55:
        return "LONG"
    if ema8 < ema13 < ema21 < ema55:
        return "SHORT"
    return "NEUTRAL"


def calculate_bollinger_bands(close: pd.Series, length: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: upper, middle, lower
    """
    bb = BollingerBands(close=close, window=length, window_dev=std)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def detect_market_structure(df: pd.DataFrame, lookback: int = 5) -> Tuple[bool, bool, str]:
    if len(df) < lookback + 2:
        return False, False, "insufficient_data"
    
    highs = df["high"].iloc[-lookback - 1:].values
    lows = df["low"].iloc[-lookback - 1:].values
    
    bullish_structure = True
    bearish_structure = True
    
    for i in range(1, len(highs)):
        if highs[i] < highs[i - 1]:
            bullish_structure = False
        if lows[i] > lows[i - 1]:
            bearish_structure = False
    
    recent_high = max(highs[-3:])
    recent_low = min(lows[-3:])
    current_price = float(df["close"].iloc[-1])
    
    hh_hl = bullish_structure and current_price > recent_low
    lh_ll = bearish_structure and current_price < recent_high
    
    if hh_hl:
        return True, False, "HH_HL_bullish"
    elif lh_ll:
        return False, True, "LH_LL_bearish"
    else:
        return False, False, "no_clear_structure"


def find_swing_high(high: pd.Series, lookback: int = 10) -> pd.Series:
    swings = pd.Series(np.nan, index=high.index)
    for i in range(lookback, len(high) - lookback):
        window = high.iloc[i - lookback : i + lookback + 1]
        if high.iloc[i] == window.max():
            swings.iloc[i] = high.iloc[i]
    return swings.ffill()


def find_swing_low(low: pd.Series, lookback: int = 10) -> pd.Series:
    swings = pd.Series(np.nan, index=low.index)
    for i in range(lookback, len(low) - lookback):
        window = low.iloc[i - lookback : i + lookback + 1]
        if low.iloc[i] == window.min():
            swings.iloc[i] = low.iloc[i]
    return swings.ffill()


def calculate_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    
    # Standard EMAs
    df["ema"] = calculate_ema(df["close"], config["ema_length"])
    df["ema7"] = calculate_ema(df["close"], config.get("ema_fast", 7))
    df["ema21"] = calculate_ema(df["close"], config.get("ema_mid", 21))
    df["ema50"] = calculate_ema(df["close"], config.get("ema_slow", 50))
    df["ema200"] = calculate_ema(df["close"], config.get("ema_trend", 200))
    
    # EMA Ribbon for scalp momentum
    df["ema8"] = calculate_ema(df["close"], 8)
    df["ema13"] = calculate_ema(df["close"], 13)
    df["ema55"] = calculate_ema(df["close"], 55)
    
    # RSI - standard and fast
    df["rsi"] = calculate_rsi(df["close"], config["rsi_length"])
    df["rsi9"] = calculate_rsi(df["close"], 9)
    
    # MACD
    macd_line, signal_line, hist_line = calculate_macd(
        df["close"], config["macd_fast"], config["macd_slow"], config["macd_signal"]
    )
    df["macd_line"] = macd_line
    df["macd_signal_line"] = signal_line
    df["macd_hist"] = hist_line
    
    # ATR
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], config["atr_length"])
    
    # Volume indicators
    if "volume" in df.columns:
        vol_s = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["vol_ma"] = calculate_volume_ma(vol_s, config["volume_length"])
        df["volume_spike"] = detect_volume_spike(vol_s, df["vol_ma"], config.get("volume_spike_mult", 1.5))
        
        # VWAP and CVD - new scalp indicators
        df["vwap"] = calculate_vwap(df["high"], df["low"], df["close"], vol_s)
        df["cvd"] = calculate_cvd(df["close"], vol_s)
    else:
        df["vol_ma"] = 0.0
        df["volume_spike"] = False
        df["vwap"] = df["close"]
        df["cvd"] = 0.0
    
    # Swing points
    df["swing_high"] = find_swing_high(df["high"], config["swing_lookback"])
    df["swing_low"] = find_swing_low(df["low"], config["swing_lookback"])
    
    # ADX
    adx, adx_pos, adx_neg = calculate_adx(df["high"], df["low"], df["close"], config.get("adx_length", 14))
    df["adx"] = adx
    df["adx_pos"] = adx_pos
    df["adx_neg"] = adx_neg
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    
    return df
