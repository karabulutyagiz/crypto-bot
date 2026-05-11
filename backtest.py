import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from indicators import calculate_all_indicators
from config import STRATEGY_CONFIG
from logger import logger

# NOTE: Bu dosya legacy backtest akışını içerir.
# Canlı stratejiye en yakın karşılaştırma/sweep için continuation_sweep.py kullanılmalıdır.


class BacktestEngine:
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.max_drawdown = 0
        self.peak_balance = initial_balance
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        risk_per_trade: float = 15,
        leverage: int = 3,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        if start_date:
            df = df[df["timestamp"] >= start_date]
        if end_date:
            df = df[df["timestamp"] <= end_date]
        
        df = calculate_all_indicators(df, STRATEGY_CONFIG)
        
        in_position = False
        current_trade = None
        
        for i in range(50, len(df)):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            if in_position and current_trade:
                current_price = candle["close"]
                
                if current_trade["side"] == "long":
                    if candle["low"] <= current_trade["stop_loss"]:
                        pnl = -risk_per_trade
                        self._close_trade(current_trade, current_trade["stop_loss"], pnl, "SL")
                        in_position = False
                    elif candle["high"] >= current_trade["tp1"] and not current_trade.get("tp1_hit"):
                        pnl = risk_per_trade * current_trade["rr"]
                        current_trade["tp1_hit"] = True
                        current_trade["stop_loss"] = current_trade["entry"]
                        current_trade["size"] *= 0.5
                    elif candle["high"] >= current_trade["tp2"] and current_trade.get("tp1_hit"):
                        remaining_pnl = (current_trade["tp2"] - current_trade["entry"]) / (current_trade["entry"] - current_trade["original_sl"]) * risk_per_trade * 0.5
                        self._close_trade(current_trade, current_trade["tp2"], remaining_pnl, "TP2")
                        in_position = False
                
                elif current_trade["side"] == "short":
                    if candle["high"] >= current_trade["stop_loss"]:
                        pnl = -risk_per_trade
                        self._close_trade(current_trade, current_trade["stop_loss"], pnl, "SL")
                        in_position = False
                    elif candle["low"] <= current_trade["tp1"] and not current_trade.get("tp1_hit"):
                        pnl = risk_per_trade * current_trade["rr"]
                        current_trade["tp1_hit"] = True
                        current_trade["stop_loss"] = current_trade["entry"]
                        current_trade["size"] *= 0.5
                    elif candle["low"] <= current_trade["tp2"] and current_trade.get("tp1_hit"):
                        remaining_pnl = (current_trade["entry"] - current_trade["tp2"]) / (current_trade["original_sl"] - current_trade["entry"]) * risk_per_trade * 0.5
                        self._close_trade(current_trade, current_trade["tp2"], remaining_pnl, "TP2")
                        in_position = False
            
            if not in_position:
                long_signal = self._check_long(df, i)
                short_signal = self._check_short(df, i)
                
                if long_signal:
                    current_trade = self._open_trade("long", candle, long_signal, risk_per_trade)
                    in_position = True
                elif short_signal:
                    current_trade = self._open_trade("short", candle, short_signal, risk_per_trade)
                    in_position = True
            
            self.equity_curve.append(self.balance)
            
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        return self._get_results()
    
    def _check_long(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        config = STRATEGY_CONFIG
        
        if candle["close"] <= candle["ema"]:
            return None
        
        if not (config["rsi_oversold"] < candle["rsi"] < 50 and candle["rsi"] > prev_candle["rsi"]):
            return None
        
        if not (candle["macd_hist"] > 0 and candle["macd_hist"] > prev_candle["macd_hist"]):
            return None
        
        if candle["volume"] <= candle["vol_ma"] * config["volume_multiplier"]:
            return None
        
        atr = candle["atr"]
        sl = min(candle["swing_low"] - atr * 0.5, candle["close"] - atr * config["atr_multiplier"]) if pd.notna(candle["swing_low"]) else candle["close"] - atr * config["atr_multiplier"]
        risk = candle["close"] - sl
        tp1 = candle["close"] + risk * config["tp1_rr"]
        tp2 = candle["close"] + risk * config["tp2_rr"]
        rr = abs(tp1 - candle["close"]) / risk
        
        if rr < config["min_rr"]:
            return None
        
        return {
            "entry": candle["close"],
            "stop_loss": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr": rr,
            "atr": atr
        }
    
    def _check_short(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        config = STRATEGY_CONFIG
        
        if candle["close"] >= candle["ema"]:
            return None
        
        if not (50 < candle["rsi"] < config["rsi_overbought"] and candle["rsi"] < prev_candle["rsi"]):
            return None
        
        if not (candle["macd_hist"] < 0 and candle["macd_hist"] < prev_candle["macd_hist"]):
            return None
        
        if candle["volume"] <= candle["vol_ma"] * config["volume_multiplier"]:
            return None
        
        atr = candle["atr"]
        sl = max(candle["swing_high"] + atr * 0.5, candle["close"] + atr * config["atr_multiplier"]) if pd.notna(candle["swing_high"]) else candle["close"] + atr * config["atr_multiplier"]
        risk = sl - candle["close"]
        tp1 = candle["close"] - risk * config["tp1_rr"]
        tp2 = candle["close"] - risk * config["tp2_rr"]
        rr = abs(tp1 - candle["close"]) / risk
        
        if rr < config["min_rr"]:
            return None
        
        return {
            "entry": candle["close"],
            "stop_loss": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr": rr,
            "atr": atr
        }
    
    def _open_trade(self, side: str, candle: pd.Series, levels: Dict, risk: float) -> Dict:
        return {
            "side": side,
            "entry": levels["entry"],
            "stop_loss": levels["stop_loss"],
            "original_sl": levels["stop_loss"],
            "tp1": levels["tp1"],
            "tp2": levels["tp2"],
            "rr": levels["rr"],
            "risk": risk,
            "size": risk / abs(levels["entry"] - levels["stop_loss"]),
            "timestamp": candle["timestamp"],
            "tp1_hit": False
        }
    
    def _close_trade(self, trade: Dict, exit_price: float, pnl: float, reason: str) -> None:
        self.balance += pnl
        self.trades.append({
            **trade,
            "exit_price": exit_price,
            "pnl": pnl,
            "close_reason": reason,
            "close_timestamp": trade["timestamp"]
        })
    
    def _get_results(self) -> Dict:
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        
        total_pnl = sum(t["pnl"] for t in self.trades)
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_return": (self.balance - self.initial_balance) / self.initial_balance * 100,
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses else float('inf'),
            "max_drawdown": self.max_drawdown,
            "trades": self.trades
        }


def run_backtest_for_symbol(df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
    logger.warning("backtest.py legacy modeldir; canlıya yakın test için continuation_sweep.py kullanın")
    engine = BacktestEngine(initial_balance=1000)
    results = engine.run_backtest(df, risk_per_trade=15, leverage=3)
    results["symbol"] = symbol
    return results
