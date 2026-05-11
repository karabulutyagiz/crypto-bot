import time
import asyncio
import signal
import os
import fcntl
import json
import threading
from html import escape
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, cast
from functools import wraps

from config import (
    SCAN_CONFIG,
    RISK_CONFIG,
    TIMEZONE_OFFSET,
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_TESTNET,
    BYBIT_DEMO_TRADING,
    ORDERBOOK_CONFIG,
    STRATEGY_CONFIG,
    EXECUTION_CONFIG,
    FLOW_TEST_MODE,
    FLOW_TEST_SYMBOL,
    FLOW_TEST_SIDE,
    FLOW_TEST_FORCE_WEAK_MODE,
    FLOW_TEST_DISABLE_NO_CHASE,
    ACTIVE_TRADING_MODE,
)
from scanner import Scanner
from strategy import analyze, get_execution_trigger
from indicators import calculate_rsi
from risk_manager import RiskManager
from telegram_bot import tg
from database import save_trade, update_trade, get_trade_history, update_daily_stats, get_open_trade
from logger import logger, log_dir, get_utc3_time
from correlation import correlation_manager
from orderbook import orderbook_analyzer
from pybit.unified_trading import HTTP


def rate_limit(max_calls: int = 10, period: float = 1.0):
    def decorator(func):
        last_calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_calls
            now = time.time()
            last_calls = [t for t in last_calls if now - t < period]
            
            if len(last_calls) >= max_calls:
                sleep_time = period - (now - last_calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            last_calls.append(time.time())
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TradingBot:
    def __init__(self):
        self._lock_handle = None
        self._lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot_instance.lock")
        if not self._acquire_instance_lock():
            raise RuntimeError("Bot zaten çalışıyor. Aynı anda ikinci instance başlatılamaz.")

        self.scanner = Scanner()
        self.risk_manager = RiskManager()
        self.running = True
        self.stop_event = threading.Event()
        self.last_trade_time: Optional[datetime] = None
        self.min_time_between_trades = timedelta(minutes=RISK_CONFIG.get("min_time_between_trades_minutes", 10))
        self.entry_timeout = timedelta(minutes=EXECUTION_CONFIG.get("entry_timeout_minutes", 30))
        self.active_positions: dict = {}
        self.pending_setups: dict = {}
        self.pending_limit_orders: dict = {}
        self.setup_requeue_tracker: dict = {}
        self.closed_positions: dict = {}
        self._missing_position_counts: dict[str, int] = {}
        self._shutdown_notified = False
        self.price_history: dict = {}
        self.restart_count = 0
        self.max_restarts = 5
        self._cached_balance: dict = {"total": 0, "available": 0, "unrealized_pnl": 0}
        self._last_balance_update: float = 0
        self._scan_cursor: int = 0
        self._state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot_state.json")
        self.trading_enabled = True

        self._load_runtime_state()
        
        self._setup_signal_handlers()
        self._setup_telegram_callbacks()

    def _acquire_instance_lock(self) -> bool:
        try:
            self._lock_handle = open(self._lock_path, "w")
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_handle.write(str(os.getpid()))
            self._lock_handle.flush()
            return True
        except Exception:
            return False

    def _release_instance_lock(self) -> None:
        try:
            if self._lock_handle:
                fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
            if os.path.exists(self._lock_path):
                os.remove(self._lock_path)
        except Exception:
            pass

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logger.info(f"Sinyal alındı: {signum}, bot durduruluyor...")
        self._notify_telegram_sync("🛑 <b>Bot kapatılıyor</b>\nSistem sinyali alındı.")
        self.running = False
        self.stop_event.set()

    def _has_open_or_pending(self) -> bool:
        if self.active_positions or self.pending_limit_orders:
            return True
        return self._check_open_positions()

    def _notify_telegram_sync(self, text: str) -> None:
        try:
            sent = tg.send_now(text)
            if not sent:
                logger.error("Telegram bildirimi gonderilemedi")
        except Exception as e:
            logger.error(f"Telegram bildirim hatası: {e}")

    def _notify_shutdown_once(self, reason: str) -> None:
        if self._shutdown_notified:
            return
        self._shutdown_notified = True
        self._notify_telegram_sync(f"🛑 <b>Bot durduruldu</b>\n{reason}")
        time.sleep(0.3)

    def _now_utc3(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=TIMEZONE_OFFSET)

    def _setup_telegram_callbacks(self) -> None:
        tg.on("status", self._get_status)
        tg.on("stop", self._stop_bot)
        tg.on("pause", self._pause_trading)
        tg.on("resume", self._resume_trading)
        tg.on("balance", self._get_balance_info)
        tg.on("history", self._get_trade_history)
        tg.on("logs", self._get_recent_logs)
        tg.setup()
        tg.start()
        logger.info("Telegram polling başlatıldı")

    def _get_status(self) -> dict:
        risk_status = self.risk_manager.get_status()
        return {
            "has_open_position": bool(self.active_positions),
            "can_trade": self.trading_enabled and risk_status.get("can_trade", False) and not bool(self.active_positions),
            "trading_enabled": self.trading_enabled,
            "pending_limit_orders": len(self.pending_limit_orders),
            "daily_losses": risk_status.get("daily_losses", 0),
            "max_daily_losses": risk_status.get("max_daily_losses", 0),
            "daily_loss_r": risk_status.get("daily_loss_r", 0),
            "max_daily_loss_r": risk_status.get("max_daily_loss_r", 0),
            "daily_trades": risk_status.get("daily_trades", 0),
            "max_daily_trades": risk_status.get("max_daily_trades", 0),
            "daily_pnl": risk_status.get("daily_pnl", 0),
            "daily_pnl_percent": risk_status.get("daily_pnl_percent", 0),
            "current_symbol": list(self.active_positions.keys())[0] if self.active_positions else None,
            "position_data": list(self.active_positions.values())[0] if self.active_positions else {}
        }

    def _load_runtime_state(self) -> None:
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.trading_enabled = bool(data.get("trading_enabled", True))
            logger.info(f"Runtime state yüklendi: trading_enabled={self.trading_enabled}")
        except Exception as e:
            logger.warning(f"Runtime state okunamadi: {e}")

    def _save_runtime_state(self) -> None:
        try:
            payload = {
                "trading_enabled": bool(self.trading_enabled),
                "updated_at": self._now_utc3().isoformat(),
            }
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning(f"Runtime state yazilamadi: {e}")

    def _check_daily_guard(self, now: datetime, balance: float) -> None:
        stop, reason = self.risk_manager.should_stop_for_day(now, balance)
        if stop:
            if self.running:
                logger.info(f"Günlük guard tetiklendi: {reason}")
                self._notify_telegram_sync(f"🛡️ <b>Günlük guard</b>\n{reason}\nBot gün sonuna kadar durduruldu.")
            self.running = False

    def _stop_bot(self) -> None:
        self._cancel_all_pending_orders("Manuel stop")
        self._notify_shutdown_once("Telegram /stop komutu")
        self.running = False
        self.stop_event.set()
        logger.info("Telegram komutu ile bot durduruldu")

    def _pause_trading(self) -> None:
        if not self.trading_enabled:
            return
        self.trading_enabled = False
        self._cancel_all_pending_orders("Trading pause")
        self._cancel_pending_setups("Trading pause")
        self._save_runtime_state()
        logger.info("Telegram komutu ile trading pause aktif")
        self._notify_telegram_sync("⏸️ <b>Trading PAUSED</b>\nYeni pozisyon açılmayacak. Açık pozisyon yönetimi devam eder.")

    def _resume_trading(self) -> None:
        if self.trading_enabled:
            return
        self.trading_enabled = True
        self._save_runtime_state()
        logger.info("Telegram komutu ile trading resume aktif")
        self._notify_telegram_sync("▶️ <b>Trading RESUMED</b>\nYeni pozisyon arama ve açma tekrar aktif.")

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        try:
            return max(1, int(str(timeframe)))
        except Exception:
            return 5

    def _get_cancel_after_candles(self, order: dict, current_price: float) -> int:
        base_candles = int(EXECUTION_CONFIG.get("cancel_after_candles", 3))
        if not EXECUTION_CONFIG.get("dynamic_cancel_candles", True):
            return max(1, base_candles)

        entry_ref = float(order.get("entry_ref", current_price) or current_price)
        atr_abs = float(order.get("atr", 0) or 0)
        if entry_ref <= 0 or atr_abs <= 0:
            return max(1, base_candles)

        atr_pct = (atr_abs / entry_ref) * 100
        high_vol_atr = float(EXECUTION_CONFIG.get("high_vol_atr_percent", 1.2))
        low_vol_atr = float(EXECUTION_CONFIG.get("low_vol_atr_percent", 0.6))

        if atr_pct >= high_vol_atr:
            return max(1, int(EXECUTION_CONFIG.get("high_vol_cancel_candles", 2)))
        if atr_pct <= low_vol_atr:
            return max(1, int(EXECUTION_CONFIG.get("low_vol_cancel_candles", 4)))
        return max(1, base_candles)

    def _cancel_all_pending_orders(self, reason: str) -> None:
        if not self.pending_limit_orders:
            return
        session = self._get_bybit_session()
        for symbol, order in list(self.pending_limit_orders.items()):
            order_id = order.get("order_id")
            if order_id:
                self._cancel_order_safe(session, symbol, str(order_id))
                logger.info(f"{symbol} bekleyen limit iptal edildi: {reason}")
            self.pending_limit_orders.pop(symbol, None)

    def _cancel_pending_setups(self, reason: str, except_symbol: Optional[str] = None) -> None:
        for sym in list(self.pending_setups.keys()):
            if except_symbol and sym == except_symbol:
                continue
            self.pending_setups.pop(sym, None)
            logger.info(f"{sym} bekleyen setup iptal edildi: {reason}")

    def _get_balance_info(self) -> dict:
        return self._cached_balance

    def _get_trade_history(self) -> list:
        return get_trade_history(limit=20)

    def _get_recent_logs(self) -> str:
        today = get_utc3_time().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"bot_{today}.log")

        if not os.path.exists(log_file):
            return "📭 Bugun icin log dosyasi bulunamadi"

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                return "📭 Log dosyasi bos"

            tail_count = 25
            recent_lines = [line.rstrip() for line in lines[-tail_count:]]
            safe_lines = "\n".join(escape(line) for line in recent_lines)
            if len(safe_lines) > 3400:
                safe_lines = safe_lines[-3400:]
                return f"📄 <b>Son loglar (kisaltilmis)</b>\n\n<code>{safe_lines}</code>"

            return f"📄 <b>Son {len(recent_lines)} log satiri</b>\n\n<code>{safe_lines}</code>"
        except Exception as e:
            return f"❌ Log okunamadi: {escape(str(e))}"

    def _build_mtf_data(self, symbol: str) -> dict:
        mtf_dfs = {}
        for tf in SCAN_CONFIG.get("higher_timeframes", []):
            tf_df = self.scanner.get_klines(symbol, interval=tf)
            if tf_df is not None and len(tf_df) >= 60:
                mtf_dfs[tf] = tf_df
        return mtf_dfs

    def _has_stale_signal_klines(
        self,
        symbol: str,
        frames: dict[str, Any],
        context: str,
        required_timeframes: Optional[set[str]] = None,
    ) -> bool:
        stale_tfs = []
        for tf, df in frames.items():
            tf_text = str(tf)
            if required_timeframes is not None and tf_text not in required_timeframes:
                continue
            if self.scanner.is_stale_kline(df):
                stale_tfs.append(tf_text)
        if not stale_tfs:
            return False
        logger.warning(f"{symbol} {context} atlandı: stale kline ({', '.join(stale_tfs)})")
        return True

    def _get_orderbook_pressure(self, symbol: str) -> tuple[float, str]:
        if not ORDERBOOK_CONFIG.get("enabled", False):
            return 0.0, "Orderbook filtresi pasif"

        pressure = orderbook_analyzer.get_market_pressure(symbol)
        if "error" in pressure:
            return 0.0, "Orderbook verisi yok"

        score = float(pressure.get("pressure_score", 0))
        sentiment = pressure.get("sentiment", "neutral")
        return score, f"Orderbook ({sentiment}, score={score:.1f})"

    def _calculate_trade_score(self, signal: str, rr: float, mtf_confirmations: int, mtf_total: int, orderbook_score: float) -> tuple[float, str]:
        base_score = 50.0
        rr_score = min(28.0, max(0.0, (rr - 1.0) * 24.0))
        mtf_score = (mtf_confirmations / mtf_total) * 20.0 if mtf_total > 0 else 0.0

        ob_weight = float(ORDERBOOK_CONFIG.get("orderbook_weight", 0.3))
        directional_orderbook = orderbook_score if signal == "LONG" else -orderbook_score
        raw_orderbook_component = directional_orderbook * ob_weight
        orderbook_component = max(-15.0, min(15.0, raw_orderbook_component))

        total_score = base_score + rr_score + mtf_score + orderbook_component
        details = (
            f"base={base_score:.1f}, rr={rr_score:.1f}, mtf={mtf_score:.1f}, "
            f"ob={orderbook_component:.1f}, total={total_score:.1f}"
        )
        return total_score, details

    def _bootstrap_existing_positions(self, positions: Optional[list] = None, notify: bool = False) -> None:
        positions = positions if positions is not None else self.scanner.get_open_positions()
        resumed_symbols = []

        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol or symbol in self.active_positions:
                continue

            db_trade = get_open_trade(symbol, side=pos.get("side"))
            pending_order = self.pending_limit_orders.get(symbol, {}) if isinstance(self.pending_limit_orders, dict) else {}
            entry_price = float(pos.get("entry_price", 0) or 0)
            size = float(pos.get("size", 0) or 0)
            stop_loss = float(
                pending_order.get("sl_trigger")
                or pos.get("stop_loss")
                or (db_trade or {}).get("stop_loss", 0)
                or 0
            )

            tp1 = float(pending_order.get("tp1", (db_trade or {}).get("take_profit1", 0)) or 0)
            tp2 = float(pending_order.get("tp2", (db_trade or {}).get("take_profit2", 0)) or 0)
            tp3 = float(pending_order.get("tp3", (db_trade or {}).get("take_profit3", 0)) or 0)
            tp1_hit = bool((db_trade or {}).get("tp1_hit", 0))
            signal = str(pending_order.get("signal") or ("LONG" if pos.get("side", "Buy") == "Buy" else "SHORT"))

            self.active_positions[symbol] = {
                "trade_id": (db_trade or {}).get("id"),
                "signal": signal,
                "side": pos.get("side", "Buy"),
                "entry_price": entry_price,
                "size": size,
                "original_size": size,
                "weak_mode": bool(pending_order.get("weak_mode", False)),
                "tp1_percent": float(pending_order.get("tp1_percent", STRATEGY_CONFIG.get("tp1_percent", 40))),
                "tp2_percent": float(pending_order.get("tp2_percent", STRATEGY_CONFIG.get("tp2_percent", 40))),
                "tp3_percent": float(pending_order.get("tp3_percent", STRATEGY_CONFIG.get("tp3_percent", 20))),
                "stop_loss": stop_loss,
                "sl_trigger": stop_loss,
                "sl_limit": float(pending_order.get("sl_limit", stop_loss) or stop_loss),
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3 if tp3 > 0 else tp2,
                "tp1_hit": tp1_hit,
                "tp1_be_applied": tp1_hit,
                "tp2_hit": False,
                "unrealised_pnl": float(pos.get("unrealised_pnl", 0) or 0),
                "realized_pnl": 0,
                "tp1_realized_pnl": 0,
                "tp2_realized_pnl": 0,
                "trailing_activated": False,
                "level_source": str(pending_order.get("level_source", "bootstrap")),
                "risk_amount": abs(entry_price - stop_loss) * size if stop_loss > 0 and entry_price > 0 else 0.0,
                "opened_at_ms": int(self._now_utc3().timestamp() * 1000),
                "entry_balance_total": self._get_balance_total_snapshot(),
            }
            correlation_manager.add_position(symbol)
            self._missing_position_counts[symbol] = 0
            resumed_symbols.append(symbol)

            if pending_order:
                self._place_position_exit_orders(symbol, self.active_positions[symbol])
                self.pending_limit_orders.pop(symbol, None)

        if self.active_positions:
            self.risk_manager.has_open_position = True
            self.risk_manager.current_symbol = next(iter(self.active_positions.keys()))

        if resumed_symbols:
            symbols_text = ", ".join(resumed_symbols)
            logger.info(f"Açık pozisyonlar senkronlandı: {symbols_text}")
            if notify:
                self._notify_telegram_sync(
                    "🔁 <b>Pozisyonlar senkronlandı</b>\n"
                    f"Exchange üzerinde açık pozisyon bulundu: {symbols_text}"
                )

    def _get_tp_plan(self, weak_mode: bool) -> dict:
        if weak_mode:
            return {
                "tp1_rr": float(STRATEGY_CONFIG.get("weak_tp1_rr", 0.9)),
                "tp2_rr": float(STRATEGY_CONFIG.get("weak_tp2_rr", 1.5)),
                "tp3_rr": float(STRATEGY_CONFIG.get("weak_tp3_rr", 2.0)),
                "tp1_percent": float(STRATEGY_CONFIG.get("weak_tp1_percent", 50)),
                "tp2_percent": float(STRATEGY_CONFIG.get("weak_tp2_percent", 40)),
                "tp3_percent": float(STRATEGY_CONFIG.get("weak_tp3_percent", 10)),
            }

        return {
            "tp1_rr": float(STRATEGY_CONFIG.get("tp1_rr", 1.0)),
            "tp2_rr": float(STRATEGY_CONFIG.get("tp2_rr", 1.7)),
            "tp3_rr": float(STRATEGY_CONFIG.get("tp3_rr", 2.5)),
            "tp1_percent": float(STRATEGY_CONFIG.get("tp1_percent", 40)),
            "tp2_percent": float(STRATEGY_CONFIG.get("tp2_percent", 40)),
            "tp3_percent": float(STRATEGY_CONFIG.get("tp3_percent", 20)),
        }

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _calculate_adaptive_sl_usdt(self, setup: dict, entry_price: float, balance: float) -> float:
        max_sl_cap = max(0.0, float(RISK_CONFIG.get("max_sl_usdt", 40.0)))
        if max_sl_cap <= 0:
            return 0.0

        base_risk = float(RISK_CONFIG.get("max_risk_per_trade", 15.0))
        fixed_fallback = float(RISK_CONFIG.get("fixed_sl_usdt", 20.0))
        adaptive_base = float(RISK_CONFIG.get("adaptive_sl_base_usdt", max(base_risk, fixed_fallback)))
        min_sl_floor = max(0.0, float(RISK_CONFIG.get("adaptive_sl_min_usdt", 10.0)))

        percent_risk = max(0.0, float(RISK_CONFIG.get("risk_percent_per_trade", 0.8)))
        balance_risk = balance * (percent_risk / 100.0) if balance > 0 else adaptive_base
        baseline = min(max_sl_cap, max(min_sl_floor, min(adaptive_base, balance_risk)))

        trade_score = float(setup.get("trade_score", 0) or 0)
        score_floor = float(RISK_CONFIG.get("adaptive_sl_score_floor", 58.0))
        score_ceil = max(score_floor + 1.0, float(RISK_CONFIG.get("adaptive_sl_score_ceiling", 78.0)))
        score_norm = self._clamp((trade_score - score_floor) / (score_ceil - score_floor), 0.0, 1.0)
        score_factor = 0.70 + (0.50 * score_norm)

        trend_strength = str(setup.get("trend_strength", "medium")).lower()
        trend_factor = 1.0
        if trend_strength == "strong":
            trend_factor = 1.08
        elif trend_strength == "weak":
            trend_factor = 0.85

        risk_multiplier = self._clamp(float(setup.get("risk_multiplier", 1.0) or 1.0), 0.6, 1.2)

        atr = float(setup.get("atr", 0) or 0)
        atr_pct = (atr / entry_price * 100.0) if atr > 0 and entry_price > 0 else 0.0
        low_vol = float(RISK_CONFIG.get("adaptive_sl_low_vol_atr_pct", 0.6))
        high_vol = max(low_vol + 0.1, float(RISK_CONFIG.get("adaptive_sl_high_vol_atr_pct", 1.8)))
        max_vol_penalty = self._clamp(float(RISK_CONFIG.get("adaptive_sl_vol_penalty_max", 0.22)), 0.0, 0.45)
        if atr_pct <= low_vol:
            vol_factor = 1.05
        elif atr_pct >= high_vol:
            vol_factor = 1.0 - max_vol_penalty
        else:
            span = high_vol - low_vol
            frac = (atr_pct - low_vol) / span if span > 0 else 0.0
            vol_factor = 1.05 - ((0.05 + max_vol_penalty) * frac)

        target_sl = baseline * score_factor * trend_factor * risk_multiplier * vol_factor

        adaptive_ceiling = float(RISK_CONFIG.get("adaptive_sl_max_usdt", max_sl_cap))
        effective_max = min(max_sl_cap, max(0.0, adaptive_ceiling))
        if effective_max <= 0:
            effective_max = max_sl_cap

        return self._clamp(target_sl, min_sl_floor, effective_max)

    def _calculate_take_profits(
        self,
        signal: str,
        entry: float,
        stop_loss: float,
        trade_score: float,
        weak_mode: bool = False,
        atr: float = 0.0,
        trend_strength: str = "medium",
        spread_bps: float = 0.0,
    ) -> tuple[float, float, float]:
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return entry, entry, entry

        plan = self._get_tp_plan(weak_mode)
        tp1_rr = max(0.1, float(plan.get("tp1_rr", 1.0)))
        base_tp2_rr = max(tp1_rr + 0.05, float(plan.get("tp2_rr", 1.7)))
        base_tp3_rr = max(base_tp2_rr + 0.05, float(plan.get("tp3_rr", 2.5)))

        min_rr_for_3tp = float(STRATEGY_CONFIG.get("min_rr_for_3tp", 2.0))
        use_single_tp = base_tp3_rr < min_rr_for_3tp

        if use_single_tp:
            single_tp_rr = max(1.5, float(STRATEGY_CONFIG.get("single_tp_rr", 1.5)))
            if signal == "LONG":
                tp1 = entry + risk * single_tp_rr
                return tp1, tp1, tp1
            else:
                tp1 = entry - risk * single_tp_rr
                return tp1, tp1, tp1

        if not bool(STRATEGY_CONFIG.get("adaptive_tp_enabled", True)):
            tp2_rr = max(base_tp2_rr, tp1_rr + 0.05)
            tp3_rr = max(base_tp3_rr, tp2_rr + 0.05)
            if signal == "LONG":
                tp1 = entry + risk * tp1_rr
                tp2 = entry + risk * tp2_rr
                tp3 = entry + risk * tp3_rr
            else:
                tp1 = entry - risk * tp1_rr
                tp2 = entry - risk * tp2_rr
                tp3 = entry - risk * tp3_rr
            return tp1, tp2, tp3

        score_floor = float(STRATEGY_CONFIG.get("adaptive_tp_score_floor", 4.0))
        score_ceil = max(score_floor + 0.5, float(STRATEGY_CONFIG.get("adaptive_tp_score_ceiling", 8.5)))
        score_norm = self._clamp((float(trade_score) - score_floor) / (score_ceil - score_floor), 0.0, 1.0)

        atr_pct = (float(atr) / float(entry) * 100.0) if atr > 0 and entry > 0 else 0.0
        low_vol = float(STRATEGY_CONFIG.get("adaptive_tp_low_vol_atr_pct", 0.60))
        high_vol = max(low_vol + 0.1, float(STRATEGY_CONFIG.get("adaptive_tp_high_vol_atr_pct", 1.80)))
        if atr_pct <= low_vol:
            vol_factor = 1.0
        elif atr_pct >= high_vol:
            vol_factor = 0.35
        else:
            vol_factor = 1.0 - (0.65 * ((atr_pct - low_vol) / (high_vol - low_vol)))

        trend_factor = 1.0
        trend_strength = str(trend_strength).lower()
        if trend_strength == "strong":
            trend_factor = 1.0
        elif trend_strength == "weak":
            trend_factor = 0.55
        else:
            trend_factor = 0.80

        spread_ref = max(1.0, float(STRATEGY_CONFIG.get("adaptive_tp_spread_ref_bps", 6.0)))
        spread_penalty_max = self._clamp(float(STRATEGY_CONFIG.get("adaptive_tp_spread_penalty_max", 0.18)), 0.0, 0.5)
        spread_ratio = self._clamp(float(spread_bps) / spread_ref, 0.0, 2.0)
        spread_factor = 1.0 - (min(1.0, spread_ratio) * spread_penalty_max)

        expansion = self._clamp(score_norm * vol_factor * trend_factor * spread_factor, 0.0, 1.0)
        tp2_bonus = float(STRATEGY_CONFIG.get("adaptive_tp2_bonus_max", 0.25)) * expansion
        tp3_bonus = float(STRATEGY_CONFIG.get("adaptive_tp3_bonus_max", 0.65)) * expansion

        tp2_rr = min(
            float(STRATEGY_CONFIG.get("adaptive_tp2_rr_max", 2.4)),
            max(base_tp2_rr, base_tp2_rr + tp2_bonus),
        )
        tp3_rr = min(
            float(STRATEGY_CONFIG.get("adaptive_tp3_rr_max", 3.0)),
            max(base_tp3_rr, base_tp3_rr + tp3_bonus),
        )

        tp2_rr = max(tp2_rr, tp1_rr + 0.10)
        tp3_rr = max(tp3_rr, tp2_rr + 0.15)

        if signal == "LONG":
            tp1 = entry + risk * tp1_rr
            tp2 = entry + risk * tp2_rr
            tp3 = entry + risk * tp3_rr
        else:
            tp1 = entry - risk * tp1_rr
            tp2 = entry - risk * tp2_rr
            tp3 = entry - risk * tp3_rr

        return tp1, tp2, tp3

    def _reanchor_levels_for_entry(
        self,
        signal: str,
        target_entry: float,
        original_entry: float,
        sl_trigger: float,
        sl_limit: float,
        tp1: float,
        tp2: float,
        tp3: float,
    ) -> tuple[float, float, float, float, float, float]:
        risk = abs(float(original_entry) - float(sl_trigger))
        if target_entry <= 0 or original_entry <= 0 or risk <= 0:
            return target_entry, sl_trigger, sl_limit, tp1, tp2, tp3

        rr1 = abs(float(tp1) - float(original_entry)) / risk if tp1 > 0 else 0.0
        rr2 = abs(float(tp2) - float(original_entry)) / risk if tp2 > 0 else 0.0
        rr3 = abs(float(tp3) - float(original_entry)) / risk if tp3 > 0 else 0.0
        sl_gap = abs(float(sl_limit) - float(sl_trigger))

        if signal == "LONG":
            new_sl_trigger = target_entry - risk
            new_sl_limit = new_sl_trigger - sl_gap
            new_tp1 = target_entry + (risk * rr1)
            new_tp2 = target_entry + (risk * rr2)
            new_tp3 = target_entry + (risk * rr3)
        else:
            new_sl_trigger = target_entry + risk
            new_sl_limit = new_sl_trigger + sl_gap
            new_tp1 = target_entry - (risk * rr1)
            new_tp2 = target_entry - (risk * rr2)
            new_tp3 = target_entry - (risk * rr3)

        return target_entry, new_sl_trigger, new_sl_limit, new_tp1, new_tp2, new_tp3

    def _try_reanchor_chased_setup(
        self,
        symbol: str,
        setup: dict,
        current_price: float,
        drift: float,
    ) -> bool:
        if not bool(EXECUTION_CONFIG.get("hard_chase_reanchor_enabled", True)):
            return False
        if bool(setup.get("chase_reanchored", False)):
            return False

        breakout_level = float(setup.get("breakout_level", 0) or 0)
        if breakout_level > 0:
            return False

        atr = float(setup.get("atr", 0) or 0)
        if atr <= 0 or current_price <= 0:
            return False

        max_reanchor_atr = float(EXECUTION_CONFIG.get("hard_chase_reanchor_max_atr", 0.90))
        min_score = float(EXECUTION_CONFIG.get("hard_chase_reanchor_min_score", 10.0))
        trade_score = float(setup.get("trade_score", 0) or 0)
        if trade_score < min_score or drift > (atr * max_reanchor_atr):
            return False

        signal = str(setup.get("signal", "")).upper()
        if signal not in ("LONG", "SHORT"):
            return False

        maker_bps = float(EXECUTION_CONFIG.get("limit_offset_bps", 1)) / 10000.0
        mode = "down" if signal == "LONG" else "up"
        target_entry = current_price * (1 - maker_bps) if signal == "LONG" else current_price * (1 + maker_bps)
        target_entry = self.scanner.normalize_price(symbol, target_entry, mode=mode)
        if target_entry <= 0:
            return False

        original_entry = float(setup.get("entry_ref", 0) or 0)
        sl_trigger = float(setup.get("sl_trigger", setup.get("stop_loss", 0)) or 0)
        sl_limit = float(setup.get("sl_limit", sl_trigger) or sl_trigger)
        tp1 = float(setup.get("est_tp1", setup.get("tp1", 0)) or 0)
        tp2 = float(setup.get("est_tp2", setup.get("tp2", 0)) or 0)
        tp3 = float(setup.get("est_tp3", setup.get("tp3", 0)) or 0)
        if original_entry <= 0 or sl_trigger <= 0:
            return False

        target_entry, new_sl_trigger, new_sl_limit, _, _, _ = self._reanchor_levels_for_entry(
            signal=signal,
            target_entry=target_entry,
            original_entry=original_entry,
            sl_trigger=sl_trigger,
            sl_limit=sl_limit,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
        )

        tp1, tp2, tp3 = self._calculate_take_profits(
            signal,
            target_entry,
            new_sl_trigger,
            trade_score,
            weak_mode=bool(setup.get("weak_mode", False)),
            atr=atr,
            trend_strength=str(setup.get("trend_strength", "medium")),
            spread_bps=float(setup.get("spread_bps", 0) or 0),
        )
        levels_ok, _ = self._validate_levels(
            symbol=symbol,
            signal=signal,
            entry=target_entry,
            sl_trigger=new_sl_trigger,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            atr=atr,
        )
        sl_ok, _ = self._validate_stop_limit(symbol, signal, new_sl_trigger, new_sl_limit)
        if not levels_ok or not sl_ok:
            return False

        old_entry = original_entry
        setup.update(
            {
                "entry_ref": target_entry,
                "limit_price": target_entry,
                "stop_loss": new_sl_trigger,
                "sl_trigger": new_sl_trigger,
                "sl_limit": new_sl_limit,
                "est_tp1": tp1,
                "est_tp2": tp2,
                "est_tp3": tp3,
                "limit_source": f"{setup.get('limit_source', 'maker_limit_on_trigger')}+hard_chase_reanchor",
                "chase_reanchored": True,
            }
        )
        logger.info(
            f"{symbol} setup re-anchor edildi: hard no-chase yerine kontrollu takip "
            f"(entry_ref {old_entry:.6f} -> {target_entry:.6f}, drift={drift:.6f}, score={trade_score:.1f})"
        )
        return True

    def _ensure_min_stop_distance(
        self,
        symbol: str,
        signal: str,
        entry: float,
        sl_trigger: float,
        sl_limit: float,
        atr: float,
    ) -> tuple[float, float]:
        tick = self.scanner.get_price_tick(symbol)
        if tick <= 0 or entry <= 0:
            return sl_trigger, sl_limit

        min_stop_atr = float(EXECUTION_CONFIG.get("min_stop_atr", 0.45))
        min_stop_ticks = max(3, int(EXECUTION_CONFIG.get("min_stop_ticks", 4)))
        min_stop = max(min_stop_atr * max(atr, 0.0), min_stop_ticks * tick)
        current_risk = abs(entry - sl_trigger)
        if current_risk >= min_stop:
            return sl_trigger, sl_limit

        sl_gap = max(abs(sl_limit - sl_trigger), 2 * tick)
        if signal == "LONG":
            new_sl_trigger = self.scanner.normalize_price(symbol, entry - min_stop, mode="down")
            new_sl_limit = self.scanner.normalize_price(symbol, new_sl_trigger - sl_gap, mode="down")
        else:
            new_sl_trigger = self.scanner.normalize_price(symbol, entry + min_stop, mode="up")
            new_sl_limit = self.scanner.normalize_price(symbol, new_sl_trigger + sl_gap, mode="up")

        logger.info(
            f"{symbol} stop mesafesi widen edildi: risk {current_risk:.10f} -> {abs(entry - new_sl_trigger):.10f}"
        )
        return new_sl_trigger, new_sl_limit

    @staticmethod
    def _levels_directionally_valid(signal: str, entry: float, sl_trigger: float, tp1: float, tp2: float, tp3: float) -> bool:
        if signal == "LONG":
            return sl_trigger < entry and tp1 > entry and tp2 > tp1 and tp3 > tp2
        return sl_trigger > entry and tp1 < entry and tp2 < tp1 and tp3 < tp2

    def _format_price(self, symbol: str, price: float) -> str:
        try:
            p = self.scanner.get_price_precision(symbol)
            return f"{float(price):.{max(0, min(10, p))}f}"
        except Exception:
            return f"{float(price):.6f}"

    def _calculate_leg_pnl(self, side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if qty <= 0 or entry_price <= 0 or exit_price <= 0:
            return 0.0
        if side == "Buy":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty

    def _get_balance_total_snapshot(self) -> float:
        try:
            balance_data = self.scanner.get_balance()
            total_balance = float(balance_data.get("total", 0) or 0)
            if total_balance > 0:
                self._cached_balance = balance_data
                self._last_balance_update = time.time()
            return total_balance
        except Exception:
            return float(self._cached_balance.get("total", 0) or 0)

    def _format_pnl(self, pnl: float) -> str:
        return f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    @staticmethod
    def _format_usd_debug(value: float) -> str:
        amount = float(value or 0.0)
        if amount >= 100:
            return f"${amount:.2f}"
        if amount >= 1:
            return f"${amount:.4f}"
        if amount > 0:
            return f"${amount:.8f}"
        return "$0.00"

    def _validate_levels(
        self,
        symbol: str,
        signal: str,
        entry: float,
        sl_trigger: float,
        tp1: float,
        tp2: float,
        tp3: float,
        atr: float,
    ) -> tuple[bool, str]:
        tick = self.scanner.get_price_tick(symbol)
        if tick <= 0:
            return False, "tick size okunamadi"

        min_ticks = 3 * tick
        min_stop_atr = float(EXECUTION_CONFIG.get("min_stop_atr", 0.45))
        min_stop_ticks = max(3, int(EXECUTION_CONFIG.get("min_stop_ticks", 4)))
        min_stop = max(min_stop_atr * max(atr, 0.0), min_stop_ticks * tick)
        risk = abs(entry - sl_trigger)

        if risk <= 0 or risk < min_ticks:
            return False, f"R gecersiz (risk={risk:.10f}, min={min_ticks:.10f})"
        if risk < min_stop:
            return False, f"stop mesafesi kucuk (risk={risk:.10f}, min_stop={min_stop:.10f})"
        if abs(tp1 - entry) < min_ticks:
            return False, "TP1 mesafesi yetersiz"

        if signal == "SHORT":
            if not (sl_trigger > entry and tp1 < entry and tp2 < tp1 and tp3 < tp2):
                return False, "SHORT seviye iliskisi gecersiz"
        else:
            if not (sl_trigger < entry and tp1 > entry and tp2 > tp1 and tp3 > tp2):
                return False, "LONG seviye iliskisi gecersiz"

        return True, f"tick={tick:g}, risk={risk:.10f}, min_stop={min_stop:.10f}"

    def _validate_stop_limit(self, symbol: str, signal: str, sl_trigger: float, sl_limit: float) -> tuple[bool, str]:
        tick = self.scanner.get_price_tick(symbol)
        if tick <= 0:
            return False, "tick size okunamadi"
        gap = abs(sl_limit - sl_trigger)
        if gap < 2 * tick:
            return False, f"SL trigger-limit farki yetersiz ({gap:.10f} < {(2*tick):.10f})"

        if signal == "LONG":
            if not (sl_limit < sl_trigger):
                return False, "LONG icin SL limit trigger altinda olmali"
        else:
            if not (sl_limit > sl_trigger):
                return False, "SHORT icin SL limit trigger ustunde olmali"

        return True, f"sl_gap={gap:.10f}"

    def _validate_qty(self, symbol: str, qty: float, entry: float) -> tuple[bool, str]:
        rules = self.scanner.get_order_rules(symbol)
        tick_size = float(rules.get("tick_size", 0) or 0)
        min_qty = float(rules.get("min_qty", 0) or 0)
        qty_step = float(rules.get("qty_step", 0) or 0)
        min_notional = float(rules.get("min_notional", 0) or 0)

        if tick_size <= 0 or qty_step <= 0 or min_qty <= 0 or min_notional <= 0:
            return False, "enstruman kurallari eksik (tick/step/minQty/minNotional)"

        if qty <= 0:
            return False, "qty sifir"
        if min_qty > 0 and qty < min_qty:
            return False, f"qty minQty altinda ({qty} < {min_qty})"
        if qty_step > 0:
            steps = round(qty / qty_step)
            if abs((steps * qty_step) - qty) > (qty_step * 1e-6):
                return False, "qty step uyumsuz"

        notional = qty * entry
        if min_notional > 0 and notional < min_notional:
            return False, f"notional min altinda ({notional:.6f} < {min_notional:.6f})"

        return True, f"qty={qty}, notional={notional:.6f}"

    def _cap_position_size_by_margin(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        available_balance: float,
    ) -> tuple[float, str]:
        if qty <= 0 or entry_price <= 0:
            return qty, "cap_skip_invalid_input"

        configured_notional_cap = max(0.0, float(RISK_CONFIG.get("max_position_size_usdt", 0.0) or 0.0))
        leverage = max(1.0, float(RISK_CONFIG.get("leverage", 1) or 1))
        margin_safety_factor = 0.95
        available_balance = max(0.0, float(available_balance or 0.0))
        margin_based_cap = available_balance * leverage * margin_safety_factor

        if configured_notional_cap > 0 and margin_based_cap > 0:
            effective_cap = min(configured_notional_cap, margin_based_cap)
        elif configured_notional_cap > 0:
            effective_cap = configured_notional_cap
        else:
            effective_cap = margin_based_cap

        planned_notional = qty * entry_price
        if effective_cap <= 0:
            return qty, (
                f"cap_disabled planned_notional={self._format_usd_debug(planned_notional)} "
                f"configured_cap={self._format_usd_debug(configured_notional_cap)} "
                f"margin_cap={self._format_usd_debug(margin_based_cap)} "
                f"available={self._format_usd_debug(available_balance)} leverage={leverage:.2f}"
            )

        if planned_notional <= effective_cap:
            return qty, (
                f"cap_ok planned_notional={self._format_usd_debug(planned_notional)} "
                f"effective_cap={self._format_usd_debug(effective_cap)}"
            )

        capped_qty = self.scanner.normalize_qty(symbol, effective_cap / entry_price)
        if capped_qty <= 0:
            return 0.0, (
                f"cap_fail planned_notional={self._format_usd_debug(planned_notional)} "
                f"effective_cap={self._format_usd_debug(effective_cap)} "
                f"available={self._format_usd_debug(available_balance)} leverage={leverage:.2f}"
            )

        return capped_qty, (
            f"cap_applied planned_notional={self._format_usd_debug(planned_notional)} "
            f"effective_cap={self._format_usd_debug(effective_cap)} "
            f"margin_cap={self._format_usd_debug(margin_based_cap)} "
            f"configured_cap={self._format_usd_debug(configured_notional_cap)}"
        )

    @staticmethod
    def _is_insufficient_margin_error(response: object) -> bool:
        if not isinstance(response, dict):
            return False
        try:
            code = int(response.get("retCode", -1))
        except Exception:
            code = -1
        msg = str(response.get("retMsg", "") or "").lower()
        return code == 110007 or "ab not enough for new order" in msg or "insufficient" in msg

    def _next_lower_qty_for_margin_retry(self, symbol: str, qty: float) -> float:
        rules = self.scanner.get_order_rules(symbol)
        qty_step = float(rules.get("qty_step", 0) or 0)
        min_qty = float(rules.get("min_qty", 0) or 0)

        reduced = qty * 0.90
        if qty_step > 0:
            reduced = max(0.0, qty - qty_step)
        reduced = self.scanner.normalize_qty(symbol, reduced)

        if reduced <= 0:
            return 0.0
        if min_qty > 0 and reduced < min_qty:
            return 0.0
        if reduced >= qty:
            return 0.0
        return reduced

    def _get_professional_limit_price(
        self,
        symbol: str,
        signal: str,
        entry_ref: float,
        df,
        atr: float,
        trend_strength: str,
    ) -> tuple[float, str]:
        offset_bps = float(EXECUTION_CONFIG.get("limit_offset_bps", 1)) / 10000
        fallback = entry_ref * (1 - offset_bps) if signal == "LONG" else entry_ref * (1 + offset_bps)

        if not EXECUTION_CONFIG.get("ema_anchor_entry", True) or df is None or len(df) < 60:
            mode = "down" if signal == "LONG" else "up"
            return self.scanner.normalize_price(symbol, fallback, mode=mode), "fallback_offset"

        close = df["close"]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        lookback = int(EXECUTION_CONFIG.get("sr_lookback_candles", 24))
        recent_window = df.iloc[-lookback - 3 : -3] if len(df) > lookback + 3 else df.iloc[:-3]
        recent_high = float(recent_window["high"].max()) if len(recent_window) > 0 else entry_ref
        recent_low = float(recent_window["low"].min()) if len(recent_window) > 0 else entry_ref

        buffer_bps = float(EXECUTION_CONFIG.get("ema_anchor_buffer_bps", 2)) / 10000
        max_dist_atr = float(EXECUTION_CONFIG.get("ema_anchor_max_distance_atr", 1.2))
        breakout_tol = float(EXECUTION_CONFIG.get("breakout_retest_tolerance_bps", 8)) / 10000
        atr_value = max(0.0, float(atr or 0))

        if signal == "LONG":
            candidates = []
            if trend_strength == "strong":
                candidates.extend([(float(ema21), "ema21")])
            elif trend_strength == "medium":
                candidates.extend([(float(ema50), "ema50")])
            else:
                candidates.extend([(float(ema200), "ema200")])
            if entry_ref >= recent_high * (1 - breakout_tol):
                candidates.append((recent_high, "breakout_retest"))

            candidates = [(a, src) for a, src in candidates if a <= entry_ref]
            if not candidates:
                return self.scanner.normalize_price(symbol, fallback, mode="down"), "fallback_offset"
            support, source = max(candidates, key=lambda item: item[0])
            if atr_value > 0 and (entry_ref - support) > atr_value * max_dist_atr:
                return self.scanner.normalize_price(symbol, fallback, mode="down"), "fallback_offset"
            anchored = support * (1 + buffer_bps)
            return self.scanner.normalize_price(symbol, anchored, mode="down"), f"{source}_support"

        candidates = []
        if trend_strength == "strong":
            candidates.extend([(float(ema21), "ema21")])
        elif trend_strength == "medium":
            candidates.extend([(float(ema50), "ema50")])
        else:
            candidates.extend([(float(ema200), "ema200")])
        if entry_ref <= recent_low * (1 + breakout_tol):
            candidates.append((recent_low, "breakout_retest"))

        candidates = [(a, src) for a, src in candidates if a >= entry_ref]
        if not candidates:
            return self.scanner.normalize_price(symbol, fallback, mode="up"), "fallback_offset"
        resistance, source = min(candidates, key=lambda item: item[0])
        if atr_value > 0 and (resistance - entry_ref) > atr_value * max_dist_atr:
            return self.scanner.normalize_price(symbol, fallback, mode="up"), "fallback_offset"
        anchored = resistance * (1 - buffer_bps)
        return self.scanner.normalize_price(symbol, anchored, mode="up"), f"{source}_resistance"

    def _attempt_pending_execution(
        self,
        symbol: str,
        setup: dict,
        now: datetime,
        balance: float,
        available_balance: float,
    ) -> bool:
        if symbol in self.pending_limit_orders:
            return False

        if self._has_open_or_pending():
            return False

        if now > setup["expires_at"]:
            logger.info(f"{symbol} setup iptal: timeout")
            self.pending_setups.pop(symbol, None)
            return False

        exec_df = self.scanner.get_klines(symbol, interval=EXECUTION_CONFIG.get("timeframe", "5"))
        if exec_df is None or len(exec_df) < 8:
            return False

        mtf_payload = self._build_mtf_data(symbol)
        if self._has_stale_signal_klines(
            symbol,
            {EXECUTION_CONFIG.get("timeframe", "5"): exec_df, **mtf_payload},
            "execution",
            required_timeframes={str(EXECUTION_CONFIG.get("timeframe", "5"))},
        ):
            return False

        if FLOW_TEST_MODE:
            triggered, trigger_reason = True, "FLOW_TEST_MODE trigger bypass"
        else:
            triggered, trigger_reason = get_execution_trigger(
                exec_df,
                setup["signal"],
                breakout_level=setup.get("breakout_level"),
                mtf_dfs=mtf_payload,
                current_time=(now.hour, now.minute, now.day, now.weekday()),
                symbol=symbol,
            )
        logger.debug(f"{symbol} execution trigger: {trigger_reason}")
        if not triggered:
            cancel_reason = ""
            try:
                close_s = cast(Any, exec_df["close"]).astype(float)
                rsi14 = float(calculate_rsi(close_s, 14).iloc[-1]) if len(close_s) >= 20 else 50.0
                ema12 = close_s.ewm(span=12, adjust=False).mean()
                ema26 = close_s.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = float((macd_line - macd_signal).iloc[-1])

                if rsi14 < 50:
                    cancel_reason = f"RSI<50 ({rsi14:.1f})"
                elif macd_hist < 0:
                    cancel_reason = f"MACD hist negatif ({macd_hist:.5f})"

                if not cancel_reason:
                    breakout_level = float(setup.get("breakout_level", 0) or 0)
                    followthrough_candles = max(1, int(EXECUTION_CONFIG.get("breakout_followthrough_candles", 2)))
                    tf_minutes = max(1, self._timeframe_to_minutes(EXECUTION_CONFIG.get("timeframe", "5")))
                    queued_at = setup.get("queued_at")
                    elapsed_candles = 0
                    if isinstance(queued_at, datetime):
                        elapsed_minutes = max(0.0, (now - queued_at).total_seconds() / 60.0)
                        elapsed_candles = int(elapsed_minutes // tf_minutes)

                    if breakout_level > 0 and elapsed_candles >= followthrough_candles and close_s.iloc[-1] <= breakout_level:
                        cancel_reason = (
                            f"breakout sonrası {followthrough_candles} mum follow-through yok "
                            f"(close={close_s.iloc[-1]:.6f} <= breakout={breakout_level:.6f})"
                        )
            except Exception:
                cancel_reason = ""

            if cancel_reason:
                logger.info(f"{symbol} setup iptal: {cancel_reason}")
                self.pending_setups.pop(symbol, None)
                return False

            logger.info(f"{symbol} strateji re-check bekleniyor: {trigger_reason}")
            return False

        current_price = self.scanner.get_current_price(symbol)
        if current_price <= 0:
            current_price = setup["entry_ref"]

        setup_atr = float(setup.get("atr", 0) or 0)
        soft_no_chase = setup_atr * float(EXECUTION_CONFIG.get("max_chase_atr", 0.55))
        hard_no_chase = setup_atr * float(EXECUTION_CONFIG.get("hard_no_chase_atr", 0.95))
        drift = abs(current_price - setup["entry_ref"])
        if (not FLOW_TEST_DISABLE_NO_CHASE) and setup_atr > 0:
            if hard_no_chase > 0 and drift > hard_no_chase:
                if self._try_reanchor_chased_setup(symbol, setup, current_price, drift):
                    current_price = float(setup.get("entry_ref", current_price) or current_price)
                    drift = abs(current_price - float(setup.get("entry_ref", current_price) or current_price))
                else:
                    logger.info(
                        f"{symbol} setup iptal: hard no-chase asildi "
                        f"(|{current_price - setup['entry_ref']:.4f}| > {hard_no_chase:.4f})"
                    )
                    self.pending_setups.pop(symbol, None)
                    return False
            if soft_no_chase > 0 and drift > soft_no_chase:
                logger.debug(
                    f"{symbol} execution beklemede: soft no-chase disinda "
                    f"(|{current_price - setup['entry_ref']:.4f}| > {soft_no_chase:.4f})"
                )
                return False

        stop_loss = setup["stop_loss"]
        sl_trigger = float(setup.get("sl_trigger", stop_loss))
        sl_limit = float(setup.get("sl_limit", sl_trigger))
        if setup["signal"] == "LONG" and current_price <= sl_trigger:
            self.pending_setups.pop(symbol, None)
            return False
        if setup["signal"] == "SHORT" and current_price >= sl_trigger:
            self.pending_setups.pop(symbol, None)
            return False

        limit_price = float(setup.get("limit_price", current_price))
        continuation_trigger = "Continuation+Triggered" in str(trigger_reason)
        if continuation_trigger:
            maker_bps = float(EXECUTION_CONFIG.get("limit_offset_bps", 1)) / 10000.0
            breakout_level = float(setup.get("breakout_level", 0) or 0)
            cont_max_chase_atr = float(EXECUTION_CONFIG.get("continuation_max_chase_atr", 0.65))
            cont_pullback_entry_atr = float(EXECUTION_CONFIG.get("continuation_pullback_entry_atr", 0.08))

            if breakout_level > 0 and setup_atr > 0 and cont_max_chase_atr > 0:
                distance_from_breakout = abs(current_price - breakout_level)
                if distance_from_breakout > (setup_atr * cont_max_chase_atr):
                    logger.info(
                        f"{symbol} continuation beklemede: fiyat breakout'tan uzak "
                        f"({distance_from_breakout:.6f}>{(setup_atr * cont_max_chase_atr):.6f})"
                    )
                    return False

            if breakout_level > 0 and setup_atr > 0:
                if setup["signal"] == "LONG":
                    anchor = breakout_level + (setup_atr * cont_pullback_entry_atr)
                    maker_price = current_price * (1 - maker_bps)
                    limit_price = min(anchor, maker_price)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="down")
                else:
                    anchor = breakout_level - (setup_atr * cont_pullback_entry_atr)
                    maker_price = current_price * (1 + maker_bps)
                    limit_price = max(anchor, maker_price)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="up")
            else:
                if setup["signal"] == "LONG":
                    limit_price = current_price * (1 - maker_bps)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="down")
                else:
                    limit_price = current_price * (1 + maker_bps)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="up")
        else:
            try:
                breakout_level = float(setup.get("breakout_level", 0) or 0)
                maker_bps = float(EXECUTION_CONFIG.get("limit_offset_bps", 1)) / 10000.0
                ema50_exec = float(exec_df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
                ema21_exec = float(exec_df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
                atr_exec = float((exec_df["high"] - exec_df["low"]).rolling(14).mean().iloc[-1] or 0)
                retest_offset = max(0.0, 0.05 * atr_exec)
                deeper_retest_offset = max(
                    retest_offset,
                    float(EXECUTION_CONFIG.get("trigger_deeper_retest_entry_atr", 0.18)) * atr_exec,
                )
                prev_low = float(exec_df.iloc[-2]["low"]) if len(exec_df) >= 2 else current_price
                prev_high = float(exec_df.iloc[-2]["high"]) if len(exec_df) >= 2 else current_price
                if setup["signal"] == "LONG":
                    prev_pullback_price = prev_low * (1 - maker_bps)
                    limit_price = min(limit_price, prev_pullback_price)
                    limit_price = min(limit_price, ema21_exec)
                    if breakout_level > 0 and ema50_exec > breakout_level:
                        ema_cap_price = ema50_exec + retest_offset
                        limit_price = min(limit_price, ema_cap_price)
                    if breakout_level > 0:
                        deeper_breakout_entry = breakout_level - deeper_retest_offset
                        limit_price = max(limit_price, deeper_breakout_entry)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="down")
                else:
                    prev_pullback_price = prev_high * (1 + maker_bps)
                    limit_price = max(limit_price, prev_pullback_price)
                    limit_price = max(limit_price, ema21_exec)
                    if breakout_level > 0 and ema50_exec < breakout_level:
                        ema_cap_price = ema50_exec - retest_offset
                        limit_price = max(limit_price, ema_cap_price)
                    if breakout_level > 0:
                        deeper_breakout_entry = breakout_level + deeper_retest_offset
                        limit_price = min(limit_price, deeper_breakout_entry)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="up")
            except Exception:
                pass
        if limit_price <= 0:
            limit_price = current_price

        breakout_level = float(setup.get("breakout_level", 0) or 0)
        if breakout_level > 0:
            order_rules = self.scanner.get_order_rules(symbol)
            tick_size = float(order_rules.get("tick_size", 0) or 0)
            tol_bps = float(
                EXECUTION_CONFIG.get(
                    "continuation_pullback_tolerance_bps",
                    EXECUTION_CONFIG.get("pullback_tolerance_bps", 3),
                )
            ) if continuation_trigger else float(EXECUTION_CONFIG.get("pullback_tolerance_bps", 3))
            zone_band = breakout_level * (tol_bps / 10000.0)
            setup_retest_band = float(setup.get("atr", 0) or 0)
            deeper_entry_band = max(
                zone_band,
                setup_retest_band * float(EXECUTION_CONFIG.get("trigger_deeper_retest_entry_atr", 0.18)),
            )
            upper_entry_band = max(
                zone_band,
                setup_retest_band * float(EXECUTION_CONFIG.get("trigger_retest_upper_band_atr", 0.08)),
            )
            if setup["signal"] == "LONG":
                zone_low = max(0.0, breakout_level - deeper_entry_band)
                zone_high = breakout_level + upper_entry_band
            else:
                zone_low = max(0.0, breakout_level - upper_entry_band)
                zone_high = breakout_level + deeper_entry_band
            zone_low_check = zone_low - tick_size if tick_size > 0 else zone_low
            zone_high_check = zone_high + tick_size if tick_size > 0 else zone_high
            if limit_price < zone_low_check or limit_price > zone_high_check:
                original_limit = limit_price
                if setup["signal"] == "LONG":
                    limit_price = min(max(limit_price, zone_low), zone_high)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="down")
                else:
                    limit_price = min(max(limit_price, zone_low), zone_high)
                    limit_price = self.scanner.normalize_price(symbol, limit_price, mode="up")

                if tick_size > 0:
                    if limit_price < zone_low and (zone_low - limit_price) <= tick_size:
                        limit_price = self.scanner.normalize_price(symbol, zone_low, mode="nearest")
                    elif limit_price > zone_high and (limit_price - zone_high) <= tick_size:
                        limit_price = self.scanner.normalize_price(symbol, zone_high, mode="nearest")

                logger.info(
                    f"{symbol} entry zone'a clamp edildi: "
                    f"{original_limit:.6f} -> {limit_price:.6f} "
                    f"(zone={zone_low:.6f}-{zone_high:.6f}, breakout={breakout_level:.6f})"
                )

                if limit_price < zone_low_check or limit_price > zone_high_check:
                    logger.info(
                        f"{symbol} setup iptal: ENTRY_REJECTED_ZONE_MISMATCH "
                        f"(limit={limit_price:.6f}, zone={zone_low:.6f}-{zone_high:.6f}, breakout={breakout_level:.6f})"
                    )
                    self.pending_setups.pop(symbol, None)
                    return False

            max_dev_pct_fixed = float(EXECUTION_CONFIG.get("max_retest_deviation_pct", 0.15))
            max_dev_atr_mult = float(EXECUTION_CONFIG.get("max_retest_deviation_atr_pct_mult", 0.25))
            atr_pct = (setup_atr / breakout_level * 100.0) if setup_atr > 0 and breakout_level > 0 else 0.0
            atr_based_dev_pct = (max_dev_atr_mult * atr_pct) if atr_pct > 0 else max_dev_pct_fixed
            max_dev_pct = min(max_dev_pct_fixed, atr_based_dev_pct)
            max_dev_abs = breakout_level * (max_dev_pct / 100.0)
            deviation_abs = abs(limit_price - breakout_level)
            if max_dev_abs > 0 and deviation_abs > max_dev_abs:
                logger.info(
                    f"{symbol} setup iptal: EXPIRED_CHASE "
                    f"(limit={limit_price:.6f}, breakout={breakout_level:.6f}, "
                    f"deviation={deviation_abs:.6f}>{max_dev_abs:.6f}, max_dev_pct={max_dev_pct:.4f}%)"
                )
                self.pending_setups.pop(symbol, None)
                return False

        tp1, tp2, tp3 = self._calculate_take_profits(
            setup["signal"],
            limit_price,
            sl_trigger,
            setup["trade_score"],
            weak_mode=bool(setup.get("weak_mode", False)),
            atr=float(setup.get("atr", 0) or 0),
            trend_strength=str(setup.get("trend_strength", "medium")),
            spread_bps=float(setup.get("spread_bps", 0) or 0),
        )

        sl_trigger, sl_limit = self._ensure_min_stop_distance(
            symbol,
            setup["signal"],
            limit_price,
            sl_trigger,
            sl_limit,
            float(setup.get("atr", 0) or 0),
        )

        tp1, tp2, tp3 = self._calculate_take_profits(
            setup["signal"],
            limit_price,
            sl_trigger,
            setup["trade_score"],
            weak_mode=bool(setup.get("weak_mode", False)),
            atr=float(setup.get("atr", 0) or 0),
            trend_strength=str(setup.get("trend_strength", "medium")),
            spread_bps=float(setup.get("spread_bps", 0) or 0),
        )

        levels_ok, levels_msg = self._validate_levels(
            symbol=symbol,
            signal=setup["signal"],
            entry=limit_price,
            sl_trigger=sl_trigger,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            atr=float(setup.get("atr", 0) or 0),
        )
        if not levels_ok:
            logger.info(f"{symbol} setup iptal: level invalid ({levels_msg})")
            self.pending_setups.pop(symbol, None)
            return False

        price_diff = abs(limit_price - sl_trigger)
        if price_diff <= 0:
            self.pending_setups.pop(symbol, None)
            return False

        adaptive_sl_enabled = bool(RISK_CONFIG.get("adaptive_sl_enabled", True))
        use_fixed_sl = bool(RISK_CONFIG.get("use_fixed_sl_usdt", False))
        position_size_mode = str(RISK_CONFIG.get("position_size_mode", "risk_based")).strip().lower()
        target_risk_usdt = 0.0
        min_risk_usdt = max(0.0, float(RISK_CONFIG.get("min_risk_per_trade", 0.0) or 0.0))
        max_risk_usdt = max(0.0, float(RISK_CONFIG.get("max_risk_per_trade", 0.0) or 0.0))
        min_notional_usdt = max(0.0, float(RISK_CONFIG.get("min_position_size_usdt", 0.0) or 0.0))
        max_notional_usdt = max(0.0, float(RISK_CONFIG.get("max_position_size_usdt", 0.0) or 0.0))

        if position_size_mode == "fixed_notional":
            position_size = self.risk_manager.calculate_position_size(limit_price, sl_trigger, balance, 1.0)
            position_size = self.scanner.normalize_qty(symbol, position_size)
            if position_size <= 0:
                logger.info(f"{symbol} setup iptal: fixed notional için qty hesaplanamadı")
                self.pending_setups.pop(symbol, None)
                return False

            fixed_notional = position_size * limit_price
            strict_risk = price_diff * position_size
            logger.info(
                f"{symbol} pozisyon boyutu fixed notional ile ayarlandı: "
                f"hedef_notional={self._format_usd_debug(max_notional_usdt)}, "
                f"uygulanacak_notional={self._format_usd_debug(fixed_notional)}, "
                f"risk={self._format_usd_debug(strict_risk)}"
            )
        elif adaptive_sl_enabled:
            target_sl_usdt = self._calculate_adaptive_sl_usdt(setup, limit_price, balance)
            target_risk_usdt = target_sl_usdt
            if target_sl_usdt <= 0:
                logger.info(f"{symbol} setup iptal: adaptive SL risk geçersiz (hedef={target_sl_usdt:.2f})")
                self.pending_setups.pop(symbol, None)
                return False

            position_size = self.scanner.normalize_qty(symbol, target_sl_usdt / price_diff)
            if position_size <= 0:
                logger.info(f"{symbol} setup iptal: adaptive SL için qty hesaplanamadı")
                self.pending_setups.pop(symbol, None)
                return False

            strict_risk = price_diff * position_size
            if strict_risk > target_sl_usdt:
                rules = self.scanner.get_order_rules(symbol)
                qty_step = float(rules.get("qty_step", 0) or 0)
                if qty_step > 0:
                    while position_size > 0 and (price_diff * position_size) > target_sl_usdt:
                        position_size = self.scanner.normalize_qty(symbol, position_size - qty_step)
                strict_risk = price_diff * position_size
                if position_size <= 0 or strict_risk > target_sl_usdt:
                    logger.info(
                        f"{symbol} setup iptal: adaptive SL risk ayarlanamadi "
                        f"(${strict_risk:.2f}>${target_sl_usdt:.2f})"
                    )
                    self.pending_setups.pop(symbol, None)
                    return False

            logger.info(
                f"{symbol} pozisyon boyutu adaptive SL ile ayarlandı: "
                f"hedef=${target_sl_usdt:.2f}, uygulanacak=${strict_risk:.2f}"
            )
        elif use_fixed_sl:
            target_sl_usdt = max(0.0, float(RISK_CONFIG.get("fixed_sl_usdt", 20.0)))
            target_sl_usdt = min(target_sl_usdt, max(0.0, float(RISK_CONFIG.get("max_sl_usdt", 40.0))))
            if max_risk_usdt > 0:
                target_sl_usdt = min(target_sl_usdt, max_risk_usdt)
            target_risk_usdt = target_sl_usdt
            if target_sl_usdt <= 0:
                logger.info(f"{symbol} setup iptal: fixed SL risk geçersiz (hedef={target_sl_usdt:.2f})")
                self.pending_setups.pop(symbol, None)
                return False

            position_size = self.scanner.normalize_qty(symbol, target_sl_usdt / price_diff)
            if position_size <= 0:
                logger.info(f"{symbol} setup iptal: sabit SL için qty hesaplanamadı")
                self.pending_setups.pop(symbol, None)
                return False

            strict_risk = price_diff * position_size
            if strict_risk > target_sl_usdt:
                rules = self.scanner.get_order_rules(symbol)
                qty_step = float(rules.get("qty_step", 0) or 0)
                if qty_step > 0:
                    while position_size > 0 and (price_diff * position_size) > target_sl_usdt:
                        position_size = self.scanner.normalize_qty(symbol, position_size - qty_step)
                strict_risk = price_diff * position_size
                if position_size <= 0 or strict_risk > target_sl_usdt:
                    logger.info(
                        f"{symbol} setup iptal: sabit SL risk ayarlanamadi "
                        f"(${strict_risk:.2f}>${target_sl_usdt:.2f})"
                    )
                    self.pending_setups.pop(symbol, None)
                    return False

            logger.info(
                f"{symbol} pozisyon boyutu sabit SL ile ayarlandı: "
                f"hedef=${target_sl_usdt:.2f}, uygulanacak=${strict_risk:.2f}"
            )
        else:
            risk_multiplier = float(setup.get("risk_multiplier", 1.0))
            position_size = self.risk_manager.calculate_position_size(limit_price, sl_trigger, balance, risk_multiplier)
            position_size = self.scanner.normalize_qty(symbol, position_size)
            if position_size <= 0:
                self.pending_setups.pop(symbol, None)
                return False

            max_sl_usdt = float(RISK_CONFIG.get("max_sl_usdt", 20.0))
            if max_sl_usdt > 0:
                planned_risk_usd = price_diff * position_size
                if planned_risk_usd > max_sl_usdt:
                    capped_size = self.scanner.normalize_qty(symbol, max_sl_usdt / price_diff)
                    if capped_size <= 0:
                        logger.info(
                            f"{symbol} setup iptal: SL risk cap asildi "
                            f"(${planned_risk_usd:.2f}>${max_sl_usdt:.2f})"
                        )
                        self.pending_setups.pop(symbol, None)
                        return False
                    logger.info(
                        f"{symbol} pozisyon boyutu risk cap ile dusuruldu: "
                        f"${planned_risk_usd:.2f} -> ${price_diff * capped_size:.2f}"
                    )
                    position_size = capped_size

                strict_risk = price_diff * position_size
                if strict_risk > max_sl_usdt:
                    rules = self.scanner.get_order_rules(symbol)
                    qty_step = float(rules.get("qty_step", 0) or 0)
                    if qty_step > 0:
                        while position_size > 0 and (price_diff * position_size) > max_sl_usdt:
                            position_size = self.scanner.normalize_qty(symbol, position_size - qty_step)
                    if position_size <= 0 or (price_diff * position_size) > max_sl_usdt:
                        logger.info(
                            f"{symbol} setup iptal: strict SL risk cap uygulanamadi "
                            f"(${price_diff * max(position_size, 0):.2f}>${max_sl_usdt:.2f})"
                        )
                        self.pending_setups.pop(symbol, None)
                        return False

        tp1_diff = abs(tp1 - limit_price)
        min_effective_sl = max(0.0, float(RISK_CONFIG.get("min_effective_sl_usdt", 0.0) or 0.0))
        min_effective_tp1 = max(0.0, float(RISK_CONFIG.get("min_effective_tp1_usdt", 0.0) or 0.0))
        min_preferred_notional = max(0.0, float(RISK_CONFIG.get("min_preferred_notional_usdt", 0.0) or 0.0))

        current_risk_usd = price_diff * position_size
        current_tp1_usd = tp1_diff * position_size
        if limit_price > 0:
            req_qty = position_size
            if min_effective_sl > 0:
                req_qty = max(req_qty, self.scanner.normalize_qty(symbol, min_effective_sl / price_diff))
            if min_effective_tp1 > 0 and tp1_diff > 0:
                req_qty = max(req_qty, self.scanner.normalize_qty(symbol, min_effective_tp1 / tp1_diff))
            if min_preferred_notional > 0:
                req_qty = max(req_qty, self.scanner.normalize_qty(symbol, min_preferred_notional / limit_price))
            if min_notional_usdt > 0:
                req_qty = max(req_qty, self.scanner.normalize_qty(symbol, min_notional_usdt / limit_price))

            if req_qty > position_size:
                boosted_risk = price_diff * req_qty
                if max_risk_usdt > 0 and boosted_risk > max_risk_usdt:
                    logger.info(
                        f"{symbol} setup iptal: min notional/efektif hedef için gereken risk fazla "
                        f"(risk=${boosted_risk:.2f} > max=${max_risk_usdt:.2f})"
                    )
                    self.pending_setups.pop(symbol, None)
                    return False
                logger.info(
                    f"{symbol} pozisyon boyutu profesyonel hedefler için artırıldı: "
                    f"qty {position_size:.8f} -> {req_qty:.8f} "
                    f"(risk ${current_risk_usd:.2f}, tp1 ${current_tp1_usd:.2f})"
                )
                position_size = req_qty

        margin_cap_applied = False
        if limit_price > 0:
            capped_qty, cap_msg = self._cap_position_size_by_margin(
                symbol,
                position_size,
                limit_price,
                available_balance,
            )
            if capped_qty <= 0:
                logger.info(f"{symbol} setup iptal: notional/margin cap uygulanamadi ({cap_msg})")
                self.pending_setups.pop(symbol, None)
                return False
            if capped_qty < position_size:
                margin_cap_applied = True
                logger.info(
                    f"{symbol} pozisyon boyutu notional/margin cap ile dusuruldu: "
                    f"${position_size * limit_price:.2f} -> ${capped_qty * limit_price:.2f} ({cap_msg})"
                )
                position_size = capped_qty

        qty_ok, qty_msg = self._validate_qty(symbol, position_size, limit_price)
        if not qty_ok:
            logger.info(f"{symbol} setup iptal: qty invalid ({qty_msg})")
            self.pending_setups.pop(symbol, None)
            return False

        risk_usd = abs(limit_price - sl_trigger) * position_size
        tp1_usd = abs(tp1 - limit_price) * position_size
        notional_usd = abs(limit_price) * position_size

        if min_notional_usdt > 0 and notional_usd < min_notional_usdt:
            logger.info(
                f"{symbol} setup iptal: notional min altinda "
                f"(${notional_usd:.2f} < ${min_notional_usdt:.2f})"
            )
            self.pending_setups.pop(symbol, None)
            return False

        if max_notional_usdt > 0 and notional_usd > max_notional_usdt:
            logger.info(
                f"{symbol} setup iptal: notional max ustunde "
                f"(${notional_usd:.2f} > ${max_notional_usdt:.2f})"
            )
            self.pending_setups.pop(symbol, None)
            return False

        if min_risk_usdt > 0 and risk_usd < min_risk_usdt:
            if use_fixed_sl and margin_cap_applied:
                capped_floor = max(
                    float(RISK_CONFIG.get("margin_capped_min_risk_floor_usdt", 2.0)),
                    target_risk_usdt * float(RISK_CONFIG.get("margin_capped_min_risk_ratio", 0.10)),
                )
                if risk_usd >= capped_floor:
                    logger.warning(
                        f"{symbol} margin cap nedeniyle dusuk risk kabul edildi "
                        f"(${risk_usd:.2f} < min=${min_risk_usdt:.2f}, capped_floor=${capped_floor:.2f})"
                    )
                else:
                    logger.info(
                        f"{symbol} setup iptal: risk min altinda "
                        f"(${risk_usd:.2f} < ${capped_floor:.2f}, margin-cap adjusted)"
                    )
                    self.pending_setups.pop(symbol, None)
                    return False
            else:
                logger.info(
                    f"{symbol} setup iptal: risk min altinda "
                    f"(${risk_usd:.2f} < ${min_risk_usdt:.2f})"
                )
                self.pending_setups.pop(symbol, None)
                return False

        if max_risk_usdt > 0 and risk_usd > max_risk_usdt:
            logger.info(
                f"{symbol} setup iptal: risk max ustunde "
                f"(${risk_usd:.2f} > ${max_risk_usdt:.2f})"
            )
            self.pending_setups.pop(symbol, None)
            return False

        if bool(RISK_CONFIG.get("enforce_min_effective_targets", True)):
            if min_effective_sl > 0 and risk_usd < min_effective_sl:
                logger.info(
                    f"{symbol} setup iptal: efektif SL hedefi tutmuyor "
                    f"(risk=${risk_usd:.2f} < min=${min_effective_sl:.2f})"
                )
                self.pending_setups.pop(symbol, None)
                return False
            if min_effective_tp1 > 0 and tp1_usd < min_effective_tp1:
                logger.info(
                    f"{symbol} setup iptal: TP1 efektif hedefi tutmuyor "
                    f"(tp1=${tp1_usd:.2f} < min=${min_effective_tp1:.2f})"
                )
                self.pending_setups.pop(symbol, None)
                return False

        if use_fixed_sl and target_risk_usdt > 0:
            min_ratio = max(0.0, float(RISK_CONFIG.get("fixed_sl_min_effective_ratio", 0.90) or 0.90))
            min_effective_floor = 0.0
            if margin_cap_applied:
                min_ratio = max(
                    0.10,
                    float(RISK_CONFIG.get("margin_capped_fixed_sl_min_effective_ratio", 0.10) or 0.10),
                )
                min_effective_floor = max(
                    2.0,
                    float(RISK_CONFIG.get("margin_capped_min_effective_sl_floor_usdt", 2.0) or 2.0),
                )
            min_effective_risk = max(target_risk_usdt * min_ratio, min_effective_floor)
            if risk_usd < min_effective_risk:
                msg = (
                    f"{symbol} margin cap ile dusuk risk kabul edildi "
                    f"(hedef=${target_risk_usdt:.2f}, efektif=${risk_usd:.2f}, min=${min_effective_risk:.2f})"
                )
                if bool(RISK_CONFIG.get("strict_fixed_sl_enforce", False)):
                    logger.info(msg)
                    self.pending_setups.pop(symbol, None)
                    return False
                logger.warning(msg)
        entry_atr = float(setup.get("atr", 0) or 0)
        risk_atr = (abs(limit_price - sl_trigger) / entry_atr) if entry_atr > 0 else 0.0
        risk_pct = (abs(limit_price - sl_trigger) / limit_price * 100) if limit_price > 0 else 0.0
        logger.info(
            f"{symbol} SL diagnostik: source={setup.get('level_source', 'n/a')}, "
            f"risk=${risk_usd:.2f}, notional=${notional_usd:.2f}, risk_pct={risk_pct:.2f}%, risk_atr={risk_atr:.2f}"
        )

        success = self._execute_trade(
            symbol,
            setup["signal"],
            limit_price,
            sl_trigger,
            sl_limit,
            tp1,
            tp2,
            tp3,
            float(setup.get("tp1_percent", self._get_tp_plan(bool(setup.get("weak_mode", False))).get("tp1_percent", 40))),
            float(setup.get("tp2_percent", self._get_tp_plan(bool(setup.get("weak_mode", False))).get("tp2_percent", 40))),
            float(setup.get("tp3_percent", self._get_tp_plan(bool(setup.get("weak_mode", False))).get("tp3_percent", 20))),
            bool(setup.get("weak_mode", False)),
            position_size,
            limit_price,
        )
        if not success:
            return False

        if symbol in self.pending_limit_orders:
            self.pending_limit_orders[symbol]["entry_ref"] = setup.get("entry_ref", limit_price)
            self.pending_limit_orders[symbol]["atr"] = setup.get("atr", 0)
            self.pending_limit_orders[symbol]["trade_score"] = setup.get("trade_score", 0)
            self.pending_limit_orders[symbol]["level_source"] = setup.get("level_source", "n/a")

        self.pending_setups.pop(symbol, None)
        return True

    def _build_setup_fingerprint(self, setup: dict) -> str:
        signal = str(setup.get("signal", "?")).upper()
        entry_ref = float(setup.get("entry_ref", 0) or 0)
        breakout_level = float(setup.get("breakout_level", 0) or 0)
        sl_trigger = float(setup.get("sl_trigger", setup.get("stop_loss", 0)) or 0)
        atr = float(setup.get("atr", 0) or 0)
        round_atr = float(EXECUTION_CONFIG.get("setup_fingerprint_round_atr", 0.35))
        base_bucket = max(atr * round_atr, entry_ref * 0.0015, 1e-8)

        def _bucketize(value: float) -> int:
            if value <= 0:
                return 0
            return int(round(value / base_bucket))

        return ":".join(
            [
                signal,
                str(_bucketize(entry_ref)),
                str(_bucketize(breakout_level)),
                str(_bucketize(sl_trigger)),
            ]
        )

    def _is_setup_requeue_blocked(self, symbol: str, setup: dict) -> bool:
        cooldown_minutes = int(EXECUTION_CONFIG.get("setup_requeue_cooldown_minutes", 15))
        if cooldown_minutes <= 0:
            return False

        now = self._now_utc3()
        tracker = self.setup_requeue_tracker.get(symbol)
        fingerprint = self._build_setup_fingerprint(setup)
        self.setup_requeue_tracker[symbol] = {
            "fingerprint": fingerprint,
            "queued_at": now,
            "signal": str(setup.get("signal", "")).upper(),
        }

        if not tracker:
            return False

        queued_at = tracker.get("queued_at")
        if not isinstance(queued_at, datetime):
            return False

        age_minutes = (now - queued_at).total_seconds() / 60.0
        if tracker.get("fingerprint") != fingerprint or age_minutes >= cooldown_minutes:
            return False

        logger.info(
            f"{symbol} setup atlandi: benzer setup cooldown aktif "
            f"({age_minutes:.1f}dk < {cooldown_minutes}dk)"
        )
        return True

    def _queue_setup(self, symbol: str, setup: dict) -> None:
        existing = self.pending_setups.get(symbol)
        if existing and existing.get("signal") == setup.get("signal"):
            return
        if self._is_setup_requeue_blocked(symbol, setup):
            return
        self.pending_setups[symbol] = setup
        trend_strength = str(setup.get("trend_strength", "n/a"))
        trend_map = {
            "strong": "🔥 strong",
            "medium": "⚖️ medium",
            "weak": "🧊 weak",
            "n/a": "❔ n/a",
        }
        trend_label = trend_map.get(trend_strength, f"❔ {trend_strength}")
        trigger_reason = str(setup.get("trigger_reason", "")).strip()
        trigger_line = f"\n<b>1M Trigger Detay:</b> {trigger_reason}" if trigger_reason else ""
        breakout_level = float(setup.get("breakout_level", 0) or 0)
        zone_low = 0.0
        zone_high = 0.0
        retest_line = ""
        if breakout_level > 0:
            tol_bps = float(EXECUTION_CONFIG.get("pullback_tolerance_bps", 3))
            zone_band = breakout_level * (tol_bps / 10000.0)
            zone_low = max(0.0, breakout_level - zone_band)
            zone_high = breakout_level + zone_band
            retest_line = (
                "\n<b>Beklenen Retest:</b> "
                f"${self._format_price(symbol, breakout_level)} "
                f"(zone ${self._format_price(symbol, zone_low)} - ${self._format_price(symbol, zone_high)})"
            )
        logger.info(
            f"{symbol} setup kuyruğa alındı ({setup['signal']}) "
            f"timeout={setup.get('timeout_minutes', EXECUTION_CONFIG.get('entry_timeout_minutes', 30))}dk"
        )
        if breakout_level > 0:
            logger.info(
                f"{symbol} beklenen retest seviyesi: {breakout_level:.6f} "
                f"(zone {zone_low:.6f}-{zone_high:.6f})"
            )
        self._notify_telegram_sync(
            f"📡 <b>Sinyal bulundu</b>\n"
            f"<b>Sembol:</b> {symbol}\n"
            f"<b>Yön:</b> {setup['signal']}\n"
            f"<b>Ref Entry:</b> ${self._format_price(symbol, setup['entry_ref'])}\n"
            f"<b>Limit:</b> ${self._format_price(symbol, setup.get('limit_price', 0))} ({setup.get('limit_source', 'n/a')})\n"
            f"<b>SL Trigger:</b> ${self._format_price(symbol, setup.get('sl_trigger', setup['stop_loss']))}\n"
            f"<b>SL Limit:</b> ${self._format_price(symbol, setup.get('sl_limit', setup['stop_loss']))}\n"
            f"<b>Tahmini TP1:</b> ${self._format_price(symbol, setup.get('est_tp1', 0))}\n"
            f"<b>Tahmini TP2:</b> ${self._format_price(symbol, setup.get('est_tp2', 0))}\n"
            f"<b>Tahmini TP3:</b> ${self._format_price(symbol, setup.get('est_tp3', 0))}\n"
            f"<b>Trend:</b> {trend_label}\n"
            f"<b>Score:</b> {setup.get('trade_score', 0):.1f}\n"
            f"<b>Durum:</b> QUEUED (1m strateji re-check bekleniyor)"
            f"{retest_line}"
            f"{trigger_line}"
        )

    def _get_setup_timeout_minutes(self, atr: float, entry: float) -> int:
        min_m = int(STRATEGY_CONFIG.get("setup_timeout_min_minutes", 15))
        max_m = int(STRATEGY_CONFIG.get("setup_timeout_max_minutes", 60))
        if atr <= 0 or entry <= 0:
            return max(min_m, min(max_m, int(EXECUTION_CONFIG.get("entry_timeout_minutes", 30))))
        atr_pct = (atr / entry) * 100
        if atr_pct >= 1.0:
            return min_m
        if atr_pct <= 0.4:
            return max_m
        return max(min_m, min(max_m, int(EXECUTION_CONFIG.get("entry_timeout_minutes", 30))))

    def _get_trigger_wait_timeout_minutes(self) -> int:
        tf_str = str(EXECUTION_CONFIG.get("timeframe", "5"))
        try:
            tf_minutes = max(1, int(tf_str))
        except Exception:
            tf_minutes = 5
        wait_candles = max(1, int(EXECUTION_CONFIG.get("trigger_wait_candles", 4)))
        return tf_minutes * wait_candles

    def _build_flow_test_setup(self, symbol: str, now: datetime) -> Optional[dict]:
        tf = str(EXECUTION_CONFIG.get("timeframe", "5"))
        df = self.scanner.get_klines(symbol, interval=tf)
        if df is None or len(df) < 40:
            return None

        current = self.scanner.get_current_price(symbol)
        if current <= 0:
            current = float(df["close"].iloc[-1])

        atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1] or 0)
        tick = self.scanner.get_price_tick(symbol)
        if current <= 0 or atr <= 0 or tick <= 0:
            return None

        signal = "SHORT" if FLOW_TEST_SIDE == "SHORT" else "LONG"
        min_stop = max(0.8 * atr, 5 * tick)
        sl_gap = max(0.2 * atr, 2 * tick)

        if signal == "LONG":
            sl_trigger = current - min_stop
            sl_limit = sl_trigger - sl_gap
            limit_price = self.scanner.normalize_price(symbol, current * 1.001, mode="up")
        else:
            sl_trigger = current + min_stop
            sl_limit = sl_trigger + sl_gap
            limit_price = self.scanner.normalize_price(symbol, current * 0.999, mode="down")

        weak_mode = bool(FLOW_TEST_FORCE_WEAK_MODE)
        est_tp1, est_tp2, est_tp3 = self._calculate_take_profits(
            signal,
            current,
            sl_trigger,
            trade_score=99.0,
            weak_mode=weak_mode,
            atr=atr,
            trend_strength="weak" if weak_mode else "medium",
            spread_bps=0.0,
        )
        levels_ok, _ = self._validate_levels(symbol, signal, current, sl_trigger, est_tp1, est_tp2, est_tp3, atr)
        sl_ok, _ = self._validate_stop_limit(symbol, signal, sl_trigger, sl_limit)
        if not levels_ok or not sl_ok:
            return None

        plan = self._get_tp_plan(weak_mode)
        return {
            "signal": signal,
            "entry_ref": current,
            "limit_price": limit_price,
            "limit_source": "flow_test_mode",
            "stop_loss": sl_trigger,
            "sl_trigger": sl_trigger,
            "sl_limit": sl_limit,
            "atr": atr,
            "trade_score": 99.0,
            "trend_strength": "weak" if weak_mode else "medium",
            "weak_mode": weak_mode,
            "weak_reason": "FLOW_TEST_MODE",
            "tp1_percent": plan.get("tp1_percent", 40),
            "tp2_percent": plan.get("tp2_percent", 40),
            "tp3_percent": plan.get("tp3_percent", 20),
            "est_tp1": est_tp1,
            "est_tp2": est_tp2,
            "est_tp3": est_tp3,
            "timeout_minutes": 15,
            "expires_at": now + timedelta(minutes=15),
        }

    def _manage_pending_limit_orders(self, now: datetime) -> None:
        if not self.pending_limit_orders:
            return

        positions = {p["symbol"]: p for p in self.scanner.get_open_positions()}
        wait_tf = self._timeframe_to_minutes(EXECUTION_CONFIG.get("cancel_wait_timeframe", "5"))
        max_chase_atr = float(EXECUTION_CONFIG.get("max_chase_atr", 0.3))
        session = self._get_bybit_session()

        for symbol, order in list(self.pending_limit_orders.items()):
            if symbol in positions and positions[symbol].get("size", 0) > 0:
                pos = positions[symbol]
                entry_price = pos.get("entry_price", order.get("entry", 0))
                size = pos.get("size", order.get("size", 0))

                trade_id = save_trade(
                    symbol=symbol,
                    side=order["side"],
                    entry_price=entry_price,
                    size=size,
                    stop_loss=order["stop_loss"],
                    tp1=order["tp1"],
                    tp2=order["tp2"],
                    tp3=order.get("tp3", order["tp2"]),
                )

                self.active_positions[symbol] = {
                    "trade_id": trade_id,
                    "signal": order.get("signal", "LONG"),
                    "side": order["side"],
                    "entry_price": entry_price,
                    "size": size,
                    "original_size": size,
                    "weak_mode": bool(order.get("weak_mode", False)),
                    "tp1_percent": float(order.get("tp1_percent", STRATEGY_CONFIG.get("tp1_percent", 40))),
                    "tp2_percent": float(order.get("tp2_percent", STRATEGY_CONFIG.get("tp2_percent", 40))),
                    "tp3_percent": float(order.get("tp3_percent", STRATEGY_CONFIG.get("tp3_percent", 20))),
                    "stop_loss": order["stop_loss"],
                    "sl_trigger": order.get("sl_trigger", order["stop_loss"]),
                    "sl_limit": order.get("sl_limit", order["stop_loss"]),
                    "tp1": order["tp1"],
                    "tp2": order["tp2"],
                    "tp3": order.get("tp3", order["tp2"]),
                    "entry_atr": float(order.get("atr", 0) or 0),
                    "level_source": str(order.get("level_source", "n/a")),
                    "risk_amount": abs(entry_price - float(order.get("sl_trigger", order["stop_loss"]))) * size,
                    "tp1_hit": False,
                    "tp1_be_applied": False,
                    "tp2_hit": False,
                    "unrealised_pnl": 0,
                    "realized_pnl": 0,
                    "tp1_realized_pnl": 0,
                    "tp2_realized_pnl": 0,
                    "trailing_activated": False,
                    "opened_at_ms": int(now.timestamp() * 1000),
                    "entry_balance_total": self._get_balance_total_snapshot(),
                }

                self._place_position_exit_orders(symbol, self.active_positions[symbol])

                if len(self.pending_limit_orders) > 1:
                    self._cancel_all_pending_orders("Tek pozisyon kuralı")

                self.risk_manager.open_position(
                    symbol,
                    signal=order.get("signal"),
                    risk_amount=self.active_positions[symbol].get("risk_amount", 0),
                )
                correlation_manager.add_position(symbol)
                self.last_trade_time = now

                self._cancel_pending_setups("Tek pozisyon açıldı", except_symbol=symbol)

                risk = abs(entry_price - order["stop_loss"])
                rr_display = abs(order["tp1"] - entry_price) / risk if risk > 0 else 0
                tg.signal_now(order["signal"], symbol, entry_price, order["stop_loss"], order["tp1"], order["tp2"], rr_display)
                self._notify_telegram_sync(
                    f"✅ <b>Pozisyon açıldı</b>\n"
                    f"<b>Sembol:</b> {symbol}\n"
                    f"<b>Yön:</b> {order['signal']}\n"
                    f"<b>Entry:</b> ${self._format_price(symbol, float(entry_price))}\n"
                    f"<b>Stop Loss:</b> ${self._format_price(symbol, float(order.get('sl_trigger', order['stop_loss'])))}\n"
                    f"<b>TP1:</b> ${self._format_price(symbol, float(order['tp1']))}\n"
                    f"<b>TP2:</b> ${self._format_price(symbol, float(order['tp2']))}\n"
                    f"<b>TP3:</b> ${self._format_price(symbol, float(order.get('tp3', order['tp2'])))}\n"
                    f"<b>Durum:</b> Limit tetiklendi, partial TP ve position SL aktif"
                )
                fill_type = "market" if order.get("market_filled") else "limit"
                risk_pct_fill = (risk / entry_price * 100) if entry_price > 0 else 0.0
                atr_fill = float(order.get("atr", 0) or 0)
                risk_atr_fill = (risk / atr_fill) if atr_fill > 0 else 0.0
                logger.info(
                    f"{symbol} {fill_type} emir doldu, pozisyon aktive edildi "
                    f"(source={order.get('level_source','n/a')}, risk=${risk * size:.2f}, "
                    f"risk_pct={risk_pct_fill:.2f}%, risk_atr={risk_atr_fill:.2f})"
                )
                self.pending_limit_orders.pop(symbol, None)
                continue

            if order.get("market_filled"):
                elapsed_market = (now - order["created_at"]).total_seconds() / 60
                if elapsed_market > 3:
                    logger.info(f"{symbol} market emir sonrası pozisyon bulunamadı, kayıt temizlendi")
                    self.pending_limit_orders.pop(symbol, None)
                continue

            elapsed_minutes = (now - order["created_at"]).total_seconds() / 60
            current_price = self.scanner.get_current_price(symbol)
            if current_price <= 0:
                current_price = float(order.get("entry_ref", order.get("entry", 0)))

            order_id = str(order.get("order_id", "") or "")
            if order_id:
                try:
                    oo = session.get_open_orders(category="linear", symbol=symbol, orderId=order_id)
                    open_list = oo.get("result", {}).get("list", []) if isinstance(oo, dict) else []
                    if not open_list:
                        logger.info(f"{symbol} bekleyen limit artık açık değil (orderId={order_id}), kayıt temizlendi")
                        self.pending_limit_orders.pop(symbol, None)
                        continue
                except Exception as e:
                    logger.debug(f"{symbol} açık emir kontrolü atlandı: {e}")

            max_wait_candles = self._get_cancel_after_candles(order, current_price)
            elapsed_candles = int(elapsed_minutes // wait_tf)

            stop_invalid = (
                (order["signal"] == "LONG" and current_price <= order["stop_loss"])
                or (order["signal"] == "SHORT" and current_price >= order["stop_loss"])
            )
            chase_invalid = (
                float(order.get("atr", 0)) > 0
                and abs(current_price - float(order.get("entry_ref", current_price))) > float(order.get("atr", 0)) * max_chase_atr
            )
            candle_timeout = elapsed_candles >= max_wait_candles

            if stop_invalid or chase_invalid or candle_timeout:
                reason = ""
                if stop_invalid:
                    reason = "fiyat stop invalidation"
                elif chase_invalid:
                    reason = "no-chase invalidation"
                else:
                    reason = (
                        f"{max_wait_candles} mum doldu "
                        f"({EXECUTION_CONFIG.get('cancel_wait_timeframe', '5')}m, dynamic)"
                    )

                order_id = order.get("order_id")
                if order_id:
                    self._cancel_order_safe(session, symbol, str(order_id))
                logger.info(f"{symbol} bekleyen limit iptal: {reason}")
                self.pending_limit_orders.pop(symbol, None)

    def _sync_correlation_state(self) -> None:
        positions = self.scanner.get_open_positions()
        exchange_symbols = {pos["symbol"] for pos in positions}

        for symbol in list(correlation_manager.active_symbols):
            if symbol not in exchange_symbols:
                correlation_manager.remove_position(symbol)

        for symbol in exchange_symbols:
            if symbol not in correlation_manager.active_symbols:
                correlation_manager.add_position(symbol)

    def _get_bybit_session(self) -> HTTP:
        return HTTP(
            testnet=BYBIT_TESTNET,
            demo=BYBIT_DEMO_TRADING,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
        )

    def _check_max_hold_time_legacy(self, symbol: str, pos_data: dict) -> None:
        """
        Pozisyonun max 1 saat tutulma süresini kontrol eder.
        1 saat dolduğunda SL'yi breakeven'a taşır (kullanıcı tercihi).
        """
        try:
            max_hold_minutes = 60  # 1 saat
            opened_at_ms = int(pos_data.get("opened_at_ms", 0) or 0)
            
            if opened_at_ms <= 0:
                return
            
            now_ms = int(self._now_utc3().timestamp() * 1000)
            hold_minutes = (now_ms - opened_at_ms) / 60000
            
            if hold_minutes < max_hold_minutes:
                return  # Henüz 1 saat dolmadı
            
            # 1 saat doldu - SL'yi breakeven'a taşı
            already_at_be = pos_data.get("sl_moved_to_be", False)
            if already_at_be:
                return  # Zaten breakeven'da
            
            entry = float(pos_data.get("entry", 0) or 0)
            if entry <= 0:
                logger.warning(f"{symbol} max hold time doldu ama entry price bulunamadı")
                return
            
            # Breakeven'a taşı
            is_long = pos_data.get("side") == "Buy"
            trigger_be = entry
            limit_be = entry * 0.998 if is_long else entry * 1.002
            
            success = self._replace_stop_limit(symbol, pos_data, trigger_be, limit_be)
            
            if success:
                pos_data["sl_moved_to_be"] = True
                pos_data["sl_be_reason"] = "max_hold_time_60min"
                logger.info(
                    f"{symbol} MAX HOLD TIME DOLDU (60 dk) - SL breakeven'a taşındı: "
                    f"entry={entry:.6f}, sl_trigger={trigger_be:.6f}"
                )
                tg.send_now(
                    f"⏱️ {symbol} 1 saat doldu\n"
                    f"SL breakeven'a taşındı: {trigger_be:.6f}\n"
                    f"Pozisyon kendi kapanmasını bekliyor."
                )
        except Exception as e:
            logger.error(f"{symbol} max hold time check hatası: {e}")
    
    @rate_limit(max_calls=5, period=1.0)
    def _execute_trade(
        self,
        symbol: str,
        signal: str,
        entry: float,
        sl_trigger: float,
        sl_limit: float,
        tp1: float,
        tp2: float,
        tp3: float,
        tp1_percent: float,
        tp2_percent: float,
        tp3_percent: float,
        weak_mode: bool,
        position_size: float,
        limit_price: float,
    ) -> bool:
        try:
            if self._has_open_or_pending():
                logger.info(f"{symbol} emir atlanıyor: açık/pending pozisyon mevcut")
                return False

            session = self._get_bybit_session()
            
            self.scanner.set_leverage(symbol, RISK_CONFIG["leverage"])
            
            side = "Buy" if signal == "LONG" else "Sell"
            order_type = str(EXECUTION_CONFIG.get("entry_order_type", "market")).lower()

            tif = "PostOnly" if EXECUTION_CONFIG.get("use_post_only", True) else "GTC"
            if FLOW_TEST_MODE:
                tif = "GTC"

            tif_to_use = tif
            if order_type == "limit" and tif == "PostOnly":
                market_price = self.scanner.get_current_price(symbol)
                if market_price > 0:
                    would_cross = (signal == "LONG" and limit_price >= market_price) or (
                        signal == "SHORT" and limit_price <= market_price
                    )
                    if would_cross:
                        tif_to_use = "GTC"
                        logger.info(
                            f"{symbol} post-only crossing engeli: tif PostOnly -> GTC "
                            f"(limit={limit_price:.6f}, last={market_price:.6f})"
                        )

            balance_data = self.scanner.get_balance()
            available_balance = float(balance_data.get("available", 0) or 0)
            if available_balance <= 0:
                available_balance = float(balance_data.get("total", 0) or 0)
            capped_qty, cap_msg = self._cap_position_size_by_margin(
                symbol,
                position_size,
                limit_price,
                available_balance,
            )
            if capped_qty <= 0:
                logger.error(f"{symbol} emir iptal: notional/margin cap sonucu qty=0 ({cap_msg})")
                return False
            if capped_qty < position_size:
                logger.info(
                    f"{symbol} emir öncesi qty düşürüldü: {position_size} -> {capped_qty} ({cap_msg})"
                )
                position_size = capped_qty

            def place_entry(qty: float, price_override: Optional[float] = None) -> dict[str, Any]:
                if order_type == "market":
                    return dict(session.place_order(
                        category="linear",
                        symbol=symbol,
                        side=side,
                        orderType="Market",
                        qty=str(qty),
                        timeInForce="IOC",
                    ))
                px = limit_price if price_override is None else float(price_override)
                return dict(session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Limit",
                    qty=str(qty),
                    price=str(px),
                    timeInForce=tif_to_use,
                ))

            response = place_entry(position_size)

            if response.get("retCode") != 0 and self._is_insufficient_margin_error(response):
                retry_balance = self.scanner.get_balance()
                retry_available = float(retry_balance.get("available", 0) or 0)
                if retry_available <= 0:
                    retry_available = float(retry_balance.get("total", 0) or 0)
                reduced_qty, retry_cap_msg = self._cap_position_size_by_margin(
                    symbol,
                    position_size,
                    limit_price,
                    retry_available,
                )
                if 0 < reduced_qty < position_size:
                    logger.warning(
                        f"{symbol} 110007 sonrası qty küçültme retry: "
                        f"{position_size} -> {reduced_qty} ({retry_cap_msg})"
                    )
                    retry_response = place_entry(reduced_qty)
                    if retry_response.get("retCode") == 0:
                        position_size = reduced_qty
                        response = retry_response
                else:
                    logger.error(
                        f"{symbol} 110007 sonrası qty küçültülemedi "
                        f"(qty={position_size}, details={retry_cap_msg})"
                    )

            if response.get("retCode") != 0 and self._is_insufficient_margin_error(response):
                max_extra_retries = 3
                current_qty = float(position_size)
                for attempt in range(1, max_extra_retries + 1):
                    next_qty = self._next_lower_qty_for_margin_retry(symbol, current_qty)
                    if next_qty <= 0:
                        break
                    logger.warning(
                        f"{symbol} margin retry #{attempt}: qty {current_qty} -> {next_qty}"
                    )
                    retry_response = place_entry(next_qty)
                    if retry_response.get("retCode") == 0:
                        position_size = next_qty
                        response = retry_response
                        break
                    if not self._is_insufficient_margin_error(retry_response):
                        response = retry_response
                        break
                    current_qty = next_qty
            
            if response.get("retCode") == 0:
                try:
                    order_id = response.get("result", {}).get("orderId", "")
                except Exception:
                    order_id = ""
                if order_type == "market":
                    logger.info(
                        f"{signal} market emir gönderildi: {symbol} qty={position_size} "
                        f"entry={limit_price:.10f}, sl={sl_trigger:.10f}, tp1={tp1:.10f}, tp2={tp2:.10f}, tp3={tp3:.10f}"
                    )
                else:
                    logger.info(
                        f"{signal} limit emir gönderildi: {symbol} @ ${entry:.4f} "
                        f"(limit={limit_price:.4f}, sl={sl_trigger:.10f}, tp1={tp1:.10f}, tp2={tp2:.10f}, tp3={tp3:.10f})"
                    )

                self.pending_limit_orders[symbol] = {
                    "order_id": order_id,
                    "signal": signal,
                    "side": side,
                    "entry": entry,
                    "entry_ref": entry,
                    "size": position_size,
                    "stop_loss": sl_trigger,
                    "sl_trigger": sl_trigger,
                    "sl_limit": sl_limit,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3,
                    "tp1_percent": tp1_percent,
                    "tp2_percent": tp2_percent,
                    "tp3_percent": tp3_percent,
                    "weak_mode": weak_mode,
                    "atr": abs(entry - sl_trigger),
                    "created_at": self._now_utc3(),
                    "market_filled": order_type == "market",
                }
                if order_type == "market":
                    self._manage_pending_limit_orders(self._now_utc3())
                return True
            else:
                if order_type == "market":
                    logger.error(f"Market emir hatası: {response.get('retMsg', 'unknown')}")
                    return False
                if EXECUTION_CONFIG.get("reprice_once", True):
                    market_price = self.scanner.get_current_price(symbol)
                    if market_price > 0:
                        reprice_bps = float(EXECUTION_CONFIG.get("reprice_offset_bps", 1)) / 10000
                        if signal == "LONG":
                            repriced = market_price * (1 - reprice_bps)
                        else:
                            repriced = market_price * (1 + reprice_bps)
                        repriced = self.scanner.normalize_price(symbol, repriced)
                        retry = dict(session.place_order(
                            category="linear",
                            symbol=symbol,
                            side=side,
                            orderType="Limit",
                            qty=str(position_size),
                            price=str(repriced),
                            timeInForce=tif_to_use,
                        ))
                        if retry.get("retCode") == 0:
                            try:
                                order_id = retry.get("result", {}).get("orderId", "")
                            except Exception:
                                order_id = ""
                            self.pending_limit_orders[symbol] = {
                                "order_id": order_id,
                                "signal": signal,
                                "side": side,
                                "entry": repriced,
                                "entry_ref": entry,
                                "size": position_size,
                                "stop_loss": sl_trigger,
                                "sl_trigger": sl_trigger,
                                "sl_limit": sl_limit,
                                "tp1": tp1,
                                "tp2": tp2,
                                "tp3": tp3,
                                "tp1_percent": tp1_percent,
                                "tp2_percent": tp2_percent,
                                "tp3_percent": tp3_percent,
                                "weak_mode": weak_mode,
                                "atr": abs(entry - sl_trigger),
                                "created_at": self._now_utc3(),
                            }
                            logger.info(f"{symbol} limit reprice ile gönderildi @ {repriced:.4f}")
                            return True
                logger.error(f"Emir hatası: {response.get('retMsg', 'unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"İşlem hatası: {e}")
            return False

    def _place_reduce_only_limit(self, session: HTTP, symbol: str, side: str, qty: float, price: float) -> Optional[str]:
        try:
            if qty <= 0 or price <= 0:
                return None
            resp = dict(session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="GTC",
                reduceOnly=True,
            ))
            if resp.get("retCode") == 0:
                return str(resp.get("result", {}).get("orderId", ""))
            logger.error(f"{symbol} TP limit emri reddedildi: {resp.get('retMsg', 'unknown')}")
        except Exception as e:
            logger.error(f"{symbol} TP limit emir hatası: {e}")
        return None

    def _set_partial_take_profit(self, session: HTTP, symbol: str, tp_price: float, tp_qty: float) -> bool:
        try:
            if tp_price <= 0 or tp_qty <= 0:
                return False

            trigger_by = "LastPrice" if EXECUTION_CONFIG.get("tp_trigger_by_last", True) else "MarkPrice"
            resp = dict(session.set_trading_stop(
                category="linear",
                symbol=symbol,
                tpslMode="Partial",
                takeProfit=str(tp_price),
                tpSize=str(tp_qty),
                tpOrderType="Market",
                tpTriggerBy=trigger_by,
                positionIdx=0,
            ))
            return bool(isinstance(resp, dict) and resp.get("retCode") == 0)
        except Exception as e:
            logger.error(f"{symbol} partial TP ayarlama hatası: {e}")
            return False

    def _place_stop_limit(self, session: HTTP, symbol: str, side: str, qty: float, trigger: float, limit_price: float) -> Optional[str]:
        try:
            if qty <= 0 or trigger <= 0 or limit_price <= 0:
                return None
            trigger_direction = 2 if side == "Sell" else 1
            resp = dict(session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(limit_price),
                triggerPrice=str(trigger),
                triggerBy="LastPrice",
                triggerDirection=trigger_direction,
                timeInForce="GTC",
                reduceOnly=True,
                closeOnTrigger=True,
            ))
            if resp.get("retCode") == 0:
                return str(resp.get("result", {}).get("orderId", ""))
        except Exception as e:
            logger.error(f"{symbol} stop-limit emir hatası: {e}")
        return None

    def _set_position_stop_loss(self, session: HTTP, symbol: str, trigger: float) -> bool:
        try:
            if trigger <= 0:
                return False
            resp = dict(session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(trigger),
                slTriggerBy="LastPrice",
                positionIdx=0,
            ))
            return bool(isinstance(resp, dict) and resp.get("retCode") == 0)
        except Exception as e:
            logger.error(f"{symbol} pozisyon SL ayarlama hatası: {e}")
            return False

    def _get_open_order_status(self, session: HTTP, symbol: str, order_id: str) -> Optional[str]:
        try:
            resp = session.get_open_orders(category="linear", symbol=symbol, orderId=order_id)
            open_list = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
            if not open_list:
                return None
            for item in open_list:
                if str(item.get("orderId", "")) == order_id:
                    status = str(item.get("orderStatus", "")).strip()
                    return status or None
            status = str(open_list[0].get("orderStatus", "")).strip()
            return status or None
        except Exception as e:
            logger.debug(f"{symbol} open-order status kontrolü atlandı ({order_id}): {e}")
            return None

    def _cancel_order_safe(self, session: HTTP, symbol: str, order_id: Optional[str]) -> None:
        if not order_id:
            return
        order_id = str(order_id)
        if order_id.startswith("POSITION_"):
            logger.debug(f"{symbol} sanal koruma emri iptal atlandı ({order_id})")
            return
        status = self._get_open_order_status(session, symbol, order_id)
        if status is None:
            logger.debug(f"{symbol} emir iptal atlandı ({order_id}): open-order kaydı yok")
            return

        blocked_statuses = {"Filled", "PartiallyFilled", "Cancelled", "Rejected", "Deactivated"}
        if status in blocked_statuses:
            logger.debug(f"{symbol} emir iptal atlandı ({order_id}): orderStatus={status}")
            return

        cancellable_statuses = {"New", "Created", "Untriggered", "Triggered", "Active"}
        if status not in cancellable_statuses:
            logger.debug(f"{symbol} emir iptal atlandı ({order_id}): bilinmeyen status={status}")
            return

        try:
            session.cancel_order(category="linear", symbol=symbol, orderId=order_id)
        except Exception as e:
            err = str(e)
            if "110001" in err or "order not exists" in err.lower() or "too late to cancel" in err.lower():
                logger.debug(f"{symbol} emir iptal atlandı ({order_id}): {err}")
                return
            logger.error(f"{symbol} emir iptal hatası ({order_id}): {e}")

    def _place_position_exit_orders(self, symbol: str, pos_data: dict) -> None:
        try:
            session = self._get_bybit_session()
            size = float(pos_data.get("original_size", pos_data.get("size", 0)) or 0)
            if size <= 0:
                return

            db_trade = get_open_trade(symbol, side=pos_data.get("side"))
            entry_price = float(pos_data.get("entry_price", 0) or 0)
            entry_atr = float(pos_data.get("entry_atr", pos_data.get("atr", 0)) or 0)
            tick_size = float(self.scanner.get_price_tick(symbol) or 0)

            tp1_pct = float(pos_data.get("tp1_percent", STRATEGY_CONFIG.get("tp1_percent", 40)))
            tp2_pct = float(pos_data.get("tp2_percent", STRATEGY_CONFIG.get("tp2_percent", 40)))
            tp3_pct = float(pos_data.get("tp3_percent", STRATEGY_CONFIG.get("tp3_percent", 20)))

            tp1_qty = self.scanner.normalize_qty(symbol, size * tp1_pct / 100)
            tp2_qty = self.scanner.normalize_qty(symbol, size * tp2_pct / 100)
            tp3_qty = self.scanner.normalize_qty(symbol, size - tp1_qty - tp2_qty)
            if tp3_qty <= 0:
                tp3_qty = self.scanner.normalize_qty(symbol, size * tp3_pct / 100)

            is_long = pos_data.get("side") == "Buy"
            signal = "LONG" if is_long else "SHORT"
            stop_loss_raw = float(
                pos_data.get("sl_trigger")
                or pos_data.get("stop_loss")
                or (db_trade or {}).get("stop_loss", 0)
                or 0
            )
            tp1_raw = float(pos_data.get("tp1") or (db_trade or {}).get("take_profit1", 0) or 0)
            tp2_raw = float(pos_data.get("tp2") or (db_trade or {}).get("take_profit2", 0) or 0)
            tp3_raw = float(pos_data.get("tp3") or (db_trade or {}).get("take_profit3", 0) or 0)

            if entry_price > 0 and (
                stop_loss_raw <= 0
                or not self._levels_directionally_valid(signal, entry_price, stop_loss_raw, tp1_raw, tp2_raw, tp3_raw)
            ):
                fallback_stop = float((db_trade or {}).get("stop_loss", 0) or 0)
                if fallback_stop > 0:
                    stop_loss_raw = fallback_stop
                if stop_loss_raw <= 0 or (is_long and stop_loss_raw >= entry_price) or ((not is_long) and stop_loss_raw <= entry_price):
                    fallback_gap = max(entry_atr * 0.8, tick_size * 5, entry_price * 0.003)
                    stop_loss_raw = entry_price - fallback_gap if is_long else entry_price + fallback_gap

                tp1_raw, tp2_raw, tp3_raw = self._calculate_take_profits(
                    signal,
                    entry_price,
                    stop_loss_raw,
                    float(pos_data.get("trade_score", 0) or 0),
                    weak_mode=bool(pos_data.get("weak_mode", False)),
                    atr=entry_atr,
                    trend_strength=str(pos_data.get("trend_strength", "medium")),
                    spread_bps=float(pos_data.get("spread_bps", 0) or 0),
                )

            sl_limit_raw = float(pos_data.get("sl_limit") or stop_loss_raw or 0)
            if sl_limit_raw <= 0 or sl_limit_raw == stop_loss_raw:
                limit_gap = max(entry_atr * 0.15, tick_size * 2, entry_price * 0.0005)
                sl_limit_raw = stop_loss_raw - limit_gap if is_long else stop_loss_raw + limit_gap

            if is_long:
                tp1_price = self.scanner.normalize_price(symbol, tp1_raw, mode="up")
                tp2_price = self.scanner.normalize_price(symbol, tp2_raw, mode="up")
                tp3_price = self.scanner.normalize_price(symbol, tp3_raw, mode="up")
                sl_trigger = self.scanner.normalize_price(symbol, stop_loss_raw, mode="down")
                sl_limit = self.scanner.normalize_price(symbol, sl_limit_raw, mode="down")
            else:
                tp1_price = self.scanner.normalize_price(symbol, tp1_raw, mode="down")
                tp2_price = self.scanner.normalize_price(symbol, tp2_raw, mode="down")
                tp3_price = self.scanner.normalize_price(symbol, tp3_raw, mode="down")
                sl_trigger = self.scanner.normalize_price(symbol, stop_loss_raw, mode="up")
                sl_limit = self.scanner.normalize_price(symbol, sl_limit_raw, mode="up")

            tp1_ok = tp1_qty <= 0 or self._set_partial_take_profit(session, symbol, tp1_price, tp1_qty)
            tp2_ok = tp2_qty <= 0 or self._set_partial_take_profit(session, symbol, tp2_price, tp2_qty)
            tp3_ok = tp3_qty <= 0 or self._set_partial_take_profit(session, symbol, tp3_price, tp3_qty)
            sl_ok = self._set_position_stop_loss(session, symbol, sl_trigger)
            sl_order = "POSITION_SL" if sl_ok else None
            if not sl_ok:
                logger.error(f"{symbol} pozisyon SL yerleşmedi")

            pos_data["tp1_qty"] = tp1_qty
            pos_data["tp2_qty"] = tp2_qty
            pos_data["tp3_qty"] = tp3_qty
            pos_data["tp1_order_id"] = "POSITION_TP1" if tp1_qty > 0 and tp1_ok else None
            pos_data["tp2_order_id"] = "POSITION_TP2" if tp2_qty > 0 and tp2_ok else None
            pos_data["tp3_order_id"] = "POSITION_TP3" if tp3_qty > 0 and tp3_ok else None
            pos_data["sl_order_id"] = sl_order
            pos_data["sl_trigger"] = sl_trigger
            pos_data["sl_limit"] = sl_limit
            pos_data["stop_loss"] = sl_trigger
            pos_data["tp1"] = tp1_price
            pos_data["tp2"] = tp2_price
            pos_data["tp3"] = tp3_price

            pos_data["sl_fallback_trading_stop"] = bool(not sl_order)

            missing_tp = int(tp1_qty > 0 and not tp1_ok) + int(tp2_qty > 0 and not tp2_ok) + int(tp3_qty > 0 and not tp3_ok)
            if missing_tp > 0:
                logger.warning(f"{symbol} TP emirlerinin bir kısmı yerleşmedi ({missing_tp}/3 eksik)")
        except Exception as e:
            logger.error(f"{symbol} exit emir yerleşimi hatası: {e}")

    def _replace_stop_limit(self, symbol: str, pos_data: dict, trigger: float, limit_price: float) -> bool:
        try:
            session = self._get_bybit_session()
            qty = self.scanner.normalize_qty(symbol, float(pos_data.get("size", 0) or 0))
            if qty <= 0:
                logger.error(f"{symbol} stop-limit güncelleme iptal: qty<=0")
                return False
            is_long = pos_data.get("side") == "Buy"
            if is_long:
                trigger_n = self.scanner.normalize_price(symbol, trigger, mode="down")
                limit_n = self.scanner.normalize_price(symbol, limit_price, mode="down")
            else:
                trigger_n = self.scanner.normalize_price(symbol, trigger, mode="up")
                limit_n = self.scanner.normalize_price(symbol, limit_price, mode="up")

            sl_ok = self._set_position_stop_loss(session, symbol, trigger_n)
            if not sl_ok:
                logger.error(f"{symbol} pozisyon SL güncellenemedi")
                return False

            pos_data["sl_order_id"] = "POSITION_SL"
            pos_data["sl_fallback_trading_stop"] = False
            pos_data["sl_trigger"] = trigger_n
            pos_data["sl_limit"] = limit_n
            pos_data["stop_loss"] = trigger_n
            logger.info(f"{symbol} position SL güncellendi: trigger={trigger_n:.6f}")
            return True
        except Exception as e:
            logger.error(f"{symbol} stop-limit güncelleme hatası: {e}")
            return False
    
    @rate_limit(max_calls=5, period=1.0)
    def _close_partial_position(self, symbol: str, close_qty: float, new_sl: float) -> bool:
        try:
            session = self._get_bybit_session()
            pos = self.active_positions.get(symbol)
            if not pos:
                return False
            
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"
            
            close_qty = self.scanner.normalize_qty(symbol, close_qty)
            if close_qty <= 0:
                return False

            response = dict(session.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=str(close_qty),
                reduceOnly=True,
            ))
            
            if response.get("retCode") == 0:
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    stopLoss=str(round(new_sl, 4)),
                    slTriggerBy="LastPrice",
                )
                logger.info(f"{symbol} kısmi kapatma: {close_qty}, yeni SL: ${new_sl:.4f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Kısmi kapatma hatası: {e}")
            return False

    def _check_max_hold_time(self, symbol: str, pos_data: dict, current_price: float) -> None:
        max_hold_minutes = float(STRATEGY_CONFIG.get("max_hold_time_minutes", 60))
        opened_at_ms = int(pos_data.get("opened_at_ms", 0) or 0)
        if opened_at_ms <= 0 or pos_data.get("hold_time_be_applied"):
            return
        now_ms = int(self._now_utc3().timestamp() * 1000)
        elapsed_minutes = (now_ms - opened_at_ms) / 60000.0
        if elapsed_minutes < max_hold_minutes:
            return
        entry_price = float(pos_data.get("entry_price", 0) or 0)
        if entry_price <= 0:
            return
        atr = float(pos_data.get("entry_atr", 0) or 0)
        be_offset = float(STRATEGY_CONFIG.get("sl_be_offset_atr", 0.0)) * max(atr, 0)
        if pos_data.get("side") == "Buy":
            be_trigger = entry_price + be_offset
            be_limit = be_trigger - (0.15 * max(atr, 0))
        else:
            be_trigger = entry_price - be_offset
            be_limit = be_trigger + (0.15 * max(atr, 0))
        current_sl = float(pos_data.get("sl_trigger", pos_data.get("stop_loss", 0)) or 0)
        if pos_data.get("side") == "Buy" and current_sl >= be_trigger:
            pos_data["hold_time_be_applied"] = True
            return
        if pos_data.get("side") == "Sell" and current_sl > 0 and current_sl <= be_trigger:
            pos_data["hold_time_be_applied"] = True
            return
        if self._replace_stop_limit(symbol, pos_data, be_trigger, be_limit):
            pos_data["hold_time_be_applied"] = True
            logger.info(
                f"{symbol} MAX HOLD TIME ({max_hold_minutes:.0f}dk) aşıldı - "
                f"SL breakeven'a taşındı: ${be_trigger:.6f} (elapsed={elapsed_minutes:.1f}dk)"
            )
            self._notify_telegram_sync(
                f"⏰ <b>MAX HOLD TIME</b>\n\n"
                f"<b>Sembol:</b> {symbol}\n"
                f"<b>Süre:</b> {elapsed_minutes:.0f} dakika\n"
                f"<b>Yeni SL:</b> BE (${be_trigger:.6f})\n"
                f"<b>Durum:</b> SL breakeven'a taşındı, kendi kapanmasını bekliyoruz"
            )
        else:
            logger.warning(f"{symbol} max hold time BE uygulanamadı, tekrar denenecek")

    def _check_open_positions(self) -> bool:
        positions = self.scanner.get_open_positions()
        return len(positions) > 0

    def _update_trailing_stop(self, symbol: str, current_price: float) -> None:
        if not RISK_CONFIG.get("trailing_stop_enabled", False):
            return

        pos_data = self.active_positions.get(symbol)
        if not pos_data:
            return

        if pos_data.get("sl_order_id"):
            return
        
        entry = pos_data["entry_price"]
        side = pos_data["side"]
        current_sl = pos_data["stop_loss"]
        
        activation_pct = RISK_CONFIG.get("trailing_stop_activation_percent", 1.0) / 100
        trail_pct = RISK_CONFIG.get("trailing_stop_percent", 1.0) / 100
        
        if side == "Buy":
            profit_pct = (current_price - entry) / entry * 100
            if profit_pct >= activation_pct * 100:
                new_sl = current_price * (1 - trail_pct)
                if new_sl > current_sl:
                    try:
                        session = self._get_bybit_session()
                        session.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            stopLoss=str(round(new_sl, 4)),
                            slTriggerBy="LastPrice",
                        )
                        pos_data["stop_loss"] = new_sl
                        pos_data["trailing_activated"] = True
                        logger.info(f"{symbol} Trailing Stop güncellendi: ${new_sl:.4f}")
                    except Exception as e:
                        logger.error(f"Trailing stop hatası: {e}")

    def _update_tp2_trailing_on_close(self, symbol: str, pos_data: dict) -> None:
        if not pos_data.get("tp2_hit"):
            return

        tf = str(EXECUTION_CONFIG.get("timeframe", "5"))
        df = self.scanner.get_klines(symbol, interval=tf)
        if df is None or len(df) < 40:
            return

        last_closed_ts = str(df.iloc[-2]["timestamp"])
        if pos_data.get("last_trail_candle_ts") == last_closed_ts:
            return

        ema21 = float(df["close"].ewm(span=21, adjust=False).mean().iloc[-2])
        atr_series = (df["high"] - df["low"]).rolling(14).mean()
        atr_now = float(atr_series.iloc[-2] if len(atr_series) >= 2 else pos_data.get("entry_atr", 0))
        atr_now = max(0.0, atr_now)

        trigger_off = float(STRATEGY_CONFIG.get("sl_trail_trigger_offset_atr", 0.10)) * atr_now
        limit_off = float(STRATEGY_CONFIG.get("sl_trail_limit_offset_atr", 0.15)) * atr_now

        if pos_data.get("side") == "Buy":
            trail_trigger = ema21 - trigger_off
            trail_limit = trail_trigger - limit_off
            if trail_trigger <= float(pos_data.get("sl_trigger", pos_data.get("stop_loss", 0))):
                pos_data["last_trail_candle_ts"] = last_closed_ts
                return
        else:
            trail_trigger = ema21 + trigger_off
            trail_limit = trail_trigger + limit_off
            if trail_trigger >= float(pos_data.get("sl_trigger", pos_data.get("stop_loss", 0))):
                pos_data["last_trail_candle_ts"] = last_closed_ts
                return

        if self._replace_stop_limit(symbol, pos_data, trail_trigger, trail_limit):
            pos_data["last_trail_candle_ts"] = last_closed_ts
            logger.info(f"{symbol} TP2 trailing SL güncellendi: trigger={trail_trigger:.6f}")
        else:
            logger.warning(f"{symbol} TP2 trailing SL güncellemesi başarısız, bir sonraki mumda tekrar denenecek")

    @rate_limit(max_calls=10, period=1.0)
    def _scan_and_trade(self) -> None:
        now = self._now_utc3()
        self._sync_correlation_state()
        
        has_position = self._check_open_positions()
        self.risk_manager.has_open_position = has_position
        collect_setups_while_open = bool(EXECUTION_CONFIG.get("collect_setups_while_open_position", True))
        can_execute_new_trades = (not has_position) and self.trading_enabled

        self._manage_pending_limit_orders(now)
        if self.pending_limit_orders:
            logger.debug("Bekleyen limit emir var, yeni setup aranmayacak")
            return

        balance_data = self.scanner.get_balance()
        balance_total = float(balance_data.get("total", 0) or 0)
        balance_available = float(balance_data.get("available", 0) or 0)
        self.risk_manager.set_day_start_balance(balance_total, now)
        self._check_daily_guard(now, balance_total)
        if not self.running:
            return

        if not self.trading_enabled:
            logger.debug("Trading pause aktif: yeni setup ve yeni emir kapalı")
            return

        if can_execute_new_trades:
            can_trade, reason = self.risk_manager.can_open_trade(now, balance_total)
            if not can_trade:
                logger.debug(reason)
                return
        else:
            if collect_setups_while_open:
                logger.debug("Açık pozisyon var: yeni emir yok, sadece setup toplanıyor")
            else:
                logger.debug("Açık pozisyon var, tarama atlanıyor...")
                return

        if can_execute_new_trades:
            for symbol in list(self.pending_setups.keys()):
                    if self._attempt_pending_execution(
                        symbol,
                        self.pending_setups[symbol],
                        now,
                        balance_total,
                        balance_available,
                    ):
                        return

        if FLOW_TEST_MODE and not self.pending_limit_orders and not self.active_positions:
            flow_symbol = FLOW_TEST_SYMBOL if FLOW_TEST_SYMBOL in self.scanner.symbols else self.scanner.symbols[0]
            if flow_symbol not in self.pending_setups:
                flow_setup = self._build_flow_test_setup(flow_symbol, now)
                if flow_setup:
                    logger.info(f"{flow_symbol} FLOW_TEST_MODE setup üretildi")
                    self._queue_setup(flow_symbol, flow_setup)
                    if self._attempt_pending_execution(
                        flow_symbol,
                        self.pending_setups[flow_symbol],
                        now,
                        balance_total,
                        balance_available,
                    ):
                        return

        if self.last_trade_time:
            if now - self.last_trade_time < self.min_time_between_trades:
                logger.debug("Minimum süre bekleniyor...")
                return

        all_symbols = list(self.scanner.symbols)
        batch_size = max(1, int(SCAN_CONFIG.get("scan_batch_size", len(all_symbols) or 1)))
        if self.scanner.is_kline_rate_limited():
            rate_limited_batch = max(1, int(SCAN_CONFIG.get("scan_batch_size_rate_limited", batch_size)))
            if rate_limited_batch < batch_size:
                batch_size = rate_limited_batch
                logger.warning(
                    f"Scanner rate-limit cooldown aktif, batch size dusuruldu: {batch_size} "
                    f"(bekleme={self.scanner.get_kline_rate_limit_remaining_seconds():.2f}s)"
                )
        if batch_size >= len(all_symbols):
            symbols_to_scan = all_symbols
        else:
            start_idx = self._scan_cursor % max(1, len(all_symbols))
            symbols_to_scan = [all_symbols[(start_idx + i) % len(all_symbols)] for i in range(batch_size)]
            self._scan_cursor = (start_idx + batch_size) % len(all_symbols)

        logger.info(
            f"{len(symbols_to_scan)}/{len(all_symbols)} sembol taranıyor"
            + (f" (batch cursor={self._scan_cursor})" if len(symbols_to_scan) < len(all_symbols) else "")
        )
        scanned_count = 0
        no_signal_count = 0
        queued_count = 0
        triggered_count = 0
        filtered_count = 0
        liquidity_filtered_count = 0
        no_signal_reasons: dict[str, int] = {}
        ticker_snapshots = self.scanner.get_ticker_snapshots()
        min_trade_turnover = float(SCAN_CONFIG.get("min_trade_turnover_usdt", 0) or 0)
        max_trade_spread_bps = float(SCAN_CONFIG.get("max_trade_spread_bps", 0) or 0)
        min_trade_price = float(SCAN_CONFIG.get("min_trade_price_usdt", 0) or 0)
        
        for symbol in symbols_to_scan:
            if not self.running:
                return
            try:
                scanned_count += 1
                can_trade_corr, corr_reason = correlation_manager.can_trade(symbol)
                if not can_trade_corr:
                    logger.debug(f"{symbol} korelasyon filtresi: {corr_reason}")
                    filtered_count += 1
                    continue

                liq = self.scanner.get_symbol_liquidity(symbol, snapshots=ticker_snapshots)
                turnover24h = float(liq.get("turnover24h", 0) or 0)
                spread_bps = float(liq.get("spread_bps", 0) or 0)
                has_book = bool(liq.get("has_book", False))
                last_price = float(liq.get("last_price", 0) or 0)

                if not has_book:
                    liquidity_filtered_count += 1
                    filtered_count += 1
                    logger.debug(f"{symbol} likidite filtresi: orderbook top-of-book yok")
                    continue
                if min_trade_turnover > 0 and turnover24h < min_trade_turnover:
                    liquidity_filtered_count += 1
                    filtered_count += 1
                    logger.debug(
                        f"{symbol} likidite filtresi: turnover24h={turnover24h:.0f} < min={min_trade_turnover:.0f}"
                    )
                    continue
                if min_trade_price > 0 and last_price < min_trade_price:
                    liquidity_filtered_count += 1
                    filtered_count += 1
                    logger.debug(
                        f"{symbol} fiyat filtresi: last={last_price:.8f} < min={min_trade_price:.8f}"
                    )
                    continue
                if max_trade_spread_bps > 0 and spread_bps > max_trade_spread_bps:
                    liquidity_filtered_count += 1
                    filtered_count += 1
                    logger.debug(
                        f"{symbol} spread filtresi: spread={spread_bps:.2f}bps > max={max_trade_spread_bps:.2f}bps"
                    )
                    continue

                entry_tf = str(EXECUTION_CONFIG.get("timeframe", "5"))
                setup_tf = "15"

                entry_df = self.scanner.get_klines(symbol, interval=entry_tf)
                if entry_df is None or len(entry_df) < 80:
                    continue

                setup_df = self.scanner.get_klines(symbol, interval=setup_tf)
                df_5m = self.scanner.get_klines(symbol, interval="5")
                df_60m = self.scanner.get_klines(symbol, interval="60")
                if setup_df is None or len(setup_df) < 80:
                    continue
                if self._has_stale_signal_klines(
                    symbol,
                    {entry_tf: entry_df, setup_tf: setup_df, "5": df_5m, "60": df_60m},
                    "scan",
                    required_timeframes={str(entry_tf), str(setup_tf)},
                ):
                    filtered_count += 1
                    continue

                current_time = self.scanner.get_current_time_utc3()
                mtf_payload = {"15": setup_df}
                if df_5m is not None:
                    mtf_payload["5"] = df_5m
                if df_60m is not None:
                    mtf_payload["60"] = df_60m
                result = analyze(
                    entry_df,
                    current_time,
                    mtf_dfs=mtf_payload,
                    symbol=symbol,
                )

                if not result or not result.get("signal"):
                    no_signal_count += 1
                    reason_text = "Analiz sonucu yok"
                    if result:
                        reason_text = str(result.get("reason", "Sinyal yok"))
                    primary_reason = reason_text.split("|", 1)[0].strip()
                    no_signal_reasons[primary_reason] = no_signal_reasons.get(primary_reason, 0) + 1
                    continue

                signal = result["signal"]
                can_dir, dir_reason = self.risk_manager.can_open_direction(signal)
                if not can_dir:
                    filtered_count += 1
                    logger.info(f"{symbol} sinyal elendi: {dir_reason}")
                    continue

                entry = float(result["entry"])
                sl = float(result["stop_loss"])
                sl_trigger = float(result.get("sl_trigger", sl))
                sl_limit = float(result.get("sl_limit", sl_trigger))
                rr = result["rr"]
                min_rr_hard = max(
                    float(STRATEGY_CONFIG.get("rr_hard_min", 1.0)),
                    float(STRATEGY_CONFIG.get("min_rr", 1.35)),
                )
                if rr < min_rr_hard:
                    filtered_count += 1
                    logger.info(f"{symbol} sinyal elendi: rr {rr:.2f} < min_rr {min_rr_hard:.2f}")
                    continue
                atr = float(result.get("atr", 0) or 0)
                trend_strength = str(result.get("trend_strength", "medium"))
                raw_quality_score = float(result.get("quality_score", result.get("total_score", 0)) or 0)
                quality_score = raw_quality_score + 1.0
                logger.info(
                    f"{symbol} state -> trend={result.get('trend_state','-')}, "
                    f"setup={result.get('setup_state','-')}, entry={result.get('entry_state','-')}, "
                    f"quality_raw={raw_quality_score:.1f}, quality_boosted={quality_score:.1f}"
                )

                weak_mode = bool(result.get("weak_mode", False))
                setup_state = str(result.get("setup_state", ""))
                breakout_level_raw = result.get("breakout_level")
                breakout_level = float(breakout_level_raw or 0)

                planned_entry_ref = float(entry)
                limit_source = "maker_limit_on_trigger"
                est_tp1 = float(result.get("tp1", 0) or 0)
                est_tp2 = float(result.get("tp2", 0) or 0)
                est_tp3 = float(result.get("tp3", 0) or 0)

                if setup_state == "QUEUED" and breakout_level > 0:
                    planned_entry_ref = breakout_level
                    limit_source = "breakout_retest_anchor"

                    invalid_anchor = (
                        (signal == "LONG" and planned_entry_ref <= sl_trigger)
                        or (signal == "SHORT" and planned_entry_ref >= sl_trigger)
                    )
                    if invalid_anchor:
                        (
                            planned_entry_ref,
                            sl_trigger,
                            sl_limit,
                            est_tp1,
                            est_tp2,
                            est_tp3,
                        ) = self._reanchor_levels_for_entry(
                            signal=signal,
                            target_entry=planned_entry_ref,
                            original_entry=entry,
                            sl_trigger=sl_trigger,
                            sl_limit=sl_limit,
                            tp1=est_tp1,
                            tp2=est_tp2,
                            tp3=est_tp3,
                        )
                        sl = sl_trigger
                        logger.info(
                            f"{symbol} queued setup seviyeleri retest entry'ye yeniden anchorlandi: "
                            f"entry={planned_entry_ref:.10f}, sl={sl_trigger:.10f}"
                        )

                    est_tp1, est_tp2, est_tp3 = self._calculate_take_profits(
                        signal,
                        planned_entry_ref,
                        sl_trigger,
                        float(quality_score),
                        weak_mode=weak_mode,
                        atr=atr,
                        trend_strength=trend_strength,
                        spread_bps=spread_bps,
                    )

                if (
                    est_tp1 <= 0
                    or est_tp2 <= 0
                    or est_tp3 <= 0
                    or not self._levels_directionally_valid(signal, planned_entry_ref, sl_trigger, est_tp1, est_tp2, est_tp3)
                ):
                    est_tp1, est_tp2, est_tp3 = self._calculate_take_profits(
                        signal,
                        planned_entry_ref,
                        sl_trigger,
                        float(quality_score),
                        weak_mode=weak_mode,
                        atr=atr,
                        trend_strength=trend_strength,
                        spread_bps=spread_bps,
                    )
                levels_ok, levels_msg = self._validate_levels(
                    symbol=symbol,
                    signal=signal,
                    entry=planned_entry_ref,
                    sl_trigger=sl_trigger,
                    tp1=est_tp1,
                    tp2=est_tp2,
                    tp3=est_tp3,
                    atr=atr,
                )
                if not levels_ok:
                    filtered_count += 1
                    logger.info(
                        f"{symbol} sinyal elendi: seviye validasyonu gecmedi ({levels_msg}) | "
                        f"signal={signal}, entry={planned_entry_ref:.10f}, sl={sl_trigger:.10f}, "
                        f"tp1={est_tp1:.10f}, tp2={est_tp2:.10f}, tp3={est_tp3:.10f}"
                    )
                    continue

                limit_price = planned_entry_ref

                setup_timeout_minutes = int(result.get("setup_timeout_minutes", self._get_setup_timeout_minutes(atr, entry)))
                min_setup_wait_minutes = max(1, int(EXECUTION_CONFIG.get("min_setup_wait_minutes", 15)))
                setup_timeout_minutes = max(
                    setup_timeout_minutes,
                    self._get_trigger_wait_timeout_minutes(),
                    min_setup_wait_minutes,
                )

                self._queue_setup(
                    symbol,
                    {
                        "signal": signal,
                        "entry_ref": planned_entry_ref,
                        "limit_price": limit_price,
                        "limit_source": limit_source,
                        "stop_loss": sl,
                        "sl_trigger": sl_trigger,
                        "sl_limit": sl_limit,
                        "atr": atr,
                        "trade_score": float(quality_score),
                        "regime": "trend",
                        "trend_strength": trend_strength,
                        "weak_mode": weak_mode,
                        "weak_reason": str(result.get("weak_reason", "")),
                        "trigger_reason": str(result.get("trigger_reason", "")),
                        "trigger_type": str(result.get("trigger_type", "")),
                        "breakout_level": breakout_level_raw,
                        "level_source": str(result.get("level_source", "n/a")),
                        "spread_bps": spread_bps,
                        "tp1_percent": float(result.get("tp1_percent", self._get_tp_plan(weak_mode).get("tp1_percent", 40))),
                        "tp2_percent": float(result.get("tp2_percent", self._get_tp_plan(weak_mode).get("tp2_percent", 40))),
                        "tp3_percent": float(result.get("tp3_percent", self._get_tp_plan(weak_mode).get("tp3_percent", 20))),
                        "est_tp1": est_tp1,
                        "est_tp2": est_tp2,
                        "est_tp3": est_tp3,
                        "timeout_minutes": setup_timeout_minutes,
                        "queued_at": now,
                        "expires_at": now + timedelta(minutes=setup_timeout_minutes),
                    },
                )

                setup_was_queued = symbol in self.pending_setups

                if str(result.get("setup_state", "")) == "QUEUED" and setup_was_queued:
                    queued_count += 1
                elif str(result.get("setup_state", "")) == "TRIGGERED" and setup_was_queued:
                    triggered_count += 1

                max_pending = max(1, int(EXECUTION_CONFIG.get("max_pending_setups", 20)))
                if len(self.pending_setups) > max_pending:
                    oldest_symbol = min(
                        self.pending_setups.keys(),
                        key=lambda s: self.pending_setups[s].get("expires_at", now),
                    )
                    if oldest_symbol != symbol:
                        self.pending_setups.pop(oldest_symbol, None)
                        logger.info(f"{oldest_symbol} setup silindi: max_pending_setups={max_pending}")

                if can_execute_new_trades and symbol in self.pending_setups:
                    if self._attempt_pending_execution(
                        symbol,
                        self.pending_setups[symbol],
                        now,
                        balance_total,
                        balance_available,
                    ):
                        return
                        
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {e}")
                continue

        top_reasons = sorted(no_signal_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
        reason_summary = " | ".join([f"{count}x {reason}" for reason, count in top_reasons]) if top_reasons else "-"
        logger.info(
            f"Tarama özeti: toplam={scanned_count}, queued={queued_count}, triggered={triggered_count}, "
            f"sinyal_yok={no_signal_count}, filtrelenen={filtered_count}, likidite_filtre={liquidity_filtered_count}, "
            f"nedenler={reason_summary}"
        )

    @rate_limit(max_calls=5, period=1.0)
    def _get_realized_pnl(self, symbol: str, since_ms: Optional[int] = None) -> tuple[float, str]:
        def _sum_closed_rows(rows: list) -> float:
            total_val = 0.0
            for row in rows:
                try:
                    total_val += float(row.get("closedPnl", 0) or 0)
                except Exception:
                    continue
            return total_val

        def _sum_exec_rows(rows: list) -> float:
            total_val = 0.0
            for row in rows:
                try:
                    total_val += float(row.get("execPnl", 0) or 0)
                except Exception:
                    continue
            return total_val

        try:
            session = self._get_bybit_session()
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": 50,
            }
            if since_ms and since_ms > 0:
                params["startTime"] = int(since_ms)

            response = dict(session.get_closed_pnl(**params))
            if response.get("retCode") != 0:
                return 0.0, "closed_pnl_unavailable"

            rows = response.get("result", {}).get("list", [])
            if not rows:
                exec_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "limit": 100,
                }
                if since_ms and since_ms > 0:
                    exec_params["startTime"] = int(since_ms)
                exec_response = dict(session.get_executions(**exec_params))
                if exec_response.get("retCode") == 0:
                    exec_rows = exec_response.get("result", {}).get("list", [])
                    exec_pnl = _sum_exec_rows(exec_rows)
                    if abs(exec_pnl) > 1e-9:
                        return exec_pnl, "execution_pnl"
                return 0.0, "no_closed_rows"

            closed_pnl = _sum_closed_rows(rows)
            if abs(closed_pnl) > 1e-9:
                return closed_pnl, "closed_pnl"

            exec_params = {
                "category": "linear",
                "symbol": symbol,
                "limit": 100,
            }
            if since_ms and since_ms > 0:
                exec_params["startTime"] = int(since_ms)
            exec_response = dict(session.get_executions(**exec_params))
            if exec_response.get("retCode") == 0:
                exec_rows = exec_response.get("result", {}).get("list", [])
                exec_pnl = _sum_exec_rows(exec_rows)
                if abs(exec_pnl) > 1e-9:
                    return exec_pnl, "execution_pnl"

            return closed_pnl, "closed_pnl_zero"
        except Exception as e:
            if since_ms and since_ms > 0:
                try:
                    session = self._get_bybit_session()
                    response = dict(session.get_closed_pnl(category="linear", symbol=symbol, limit=50))
                    if response.get("retCode") == 0:
                        rows = response.get("result", {}).get("list", [])
                        closed_pnl = _sum_closed_rows(rows)
                        if abs(closed_pnl) > 1e-9:
                            return closed_pnl, "closed_pnl_fallback"
                    exec_response = dict(session.get_executions(category="linear", symbol=symbol, limit=100))
                    if exec_response.get("retCode") == 0:
                        exec_rows = exec_response.get("result", {}).get("list", [])
                        exec_pnl = _sum_exec_rows(exec_rows)
                        if abs(exec_pnl) > 1e-9:
                            return exec_pnl, "execution_pnl_fallback"
                except Exception:
                    pass
            logger.error(f"Realized PnL hatası: {e}")
            return 0.0, "error"

    def _monitor_positions(self) -> None:
        if not self.running:
            return
        self._sync_correlation_state()
        positions = self.scanner.get_open_positions()
        if self.active_positions and not positions:
            logger.warning("Pozisyon listesi boş döndü; olası API hatası nedeniyle close işlemleri bu tur atlandı")
            return
        self._bootstrap_existing_positions(positions=positions, notify=False)
        current_symbols = {pos["symbol"] for pos in positions}
        missing_confirmations_required = max(2, int(RISK_CONFIG.get("position_missing_confirmations", 2)))
        force_close_confirmations = max(
            missing_confirmations_required + 1,
            int(RISK_CONFIG.get("position_force_close_confirmations", 4)),
        )
        
        for symbol in list(self.active_positions.keys()):
            pos_data = self.active_positions[symbol]
            
            if symbol not in current_symbols:
                miss_count = self._missing_position_counts.get(symbol, 0) + 1
                self._missing_position_counts[symbol] = miss_count
                if miss_count < missing_confirmations_required:
                    logger.warning(
                        f"{symbol} pozisyonu exchange listesinde görünmedi "
                        f"({miss_count}/{missing_confirmations_required}); close onayı bekleniyor"
                    )
                    continue

                opened_at_ms = int(pos_data.get("opened_at_ms", 0) or 0)
                realized_pnl, pnl_source = self._get_realized_pnl(symbol, since_ms=opened_at_ms)
                if abs(realized_pnl) <= 1e-9:
                    last_unrealized = float(pos_data.get("unrealised_pnl", 0) or 0)
                    unreliable_sources = {"closed_pnl_unavailable", "no_closed_rows", "closed_pnl_zero", "error"}
                    if abs(last_unrealized) > 1e-9:
                        realized_pnl = last_unrealized
                        pnl_source = "last_unrealized_snapshot"
                    elif pnl_source in unreliable_sources and miss_count < force_close_confirmations:
                        logger.warning(
                            f"{symbol} close doğrulanamadı; PnL kaynağı güvenilmez ({pnl_source}) "
                            f"ve miss_count={miss_count}/{force_close_confirmations}. Takip sürüyor"
                        )
                        continue

                entry_balance_total = float(pos_data.get("entry_balance_total", 0) or 0)
                exit_balance_total = self._get_balance_total_snapshot()
                balance_delta_valid = entry_balance_total > 0 and exit_balance_total > 0
                effective_pnl = realized_pnl

                was_profit = effective_pnl >= 0
                prior_realized = float(pos_data.get("realized_pnl", 0) or 0)
                final_leg_pnl = effective_pnl - prior_realized
                
                self.risk_manager.close_position(
                    was_profit,
                    effective_pnl,
                    self._now_utc3(),
                    signal=pos_data.get("signal"),
                    risk_amount=float(pos_data.get("risk_amount", 0) or 0),
                )

                session = self._get_bybit_session()
                self._cancel_order_safe(session, symbol, pos_data.get("tp1_order_id"))
                self._cancel_order_safe(session, symbol, pos_data.get("tp2_order_id"))
                self._cancel_order_safe(session, symbol, pos_data.get("tp3_order_id"))
                self._cancel_order_safe(session, symbol, pos_data.get("sl_order_id"))
                
                if pos_data.get("trade_id"):
                    update_trade(
                        trade_id=pos_data["trade_id"],
                        pnl=effective_pnl,
                        status="closed"
                    )
                
                today = self._now_utc3().strftime("%Y-%m-%d")
                update_daily_stats(today, effective_pnl, was_profit)
                
                tg.closed_now(
                    sym=symbol,
                    side=pos_data["side"],
                    pnl=effective_pnl,
                    leg_pnl=final_leg_pnl,
                    cumulative_pnl=effective_pnl,
                    balance_before=entry_balance_total if balance_delta_valid else None,
                    balance_after=exit_balance_total if balance_delta_valid else None,
                    reason=(
                        "TP/SL"
                        if pnl_source.startswith("closed_pnl") or pnl_source.startswith("execution_pnl")
                        else "Manuel/Snapshot"
                    ),
                )
                logger.info(
                    f"{symbol} pozisyon kapandı - PnL: ${effective_pnl:.2f} "
                    f"(source={pnl_source}, raw_realized=${realized_pnl:.2f}, balance_before=${entry_balance_total:.2f}, balance_after=${exit_balance_total:.2f})"
                )
                
                self.closed_positions[symbol] = {
                    **pos_data,
                    "realized_pnl": effective_pnl,
                    "raw_realized_pnl": realized_pnl,
                    "entry_balance_total": entry_balance_total,
                    "exit_balance_total": exit_balance_total,
                    "close_time": self._now_utc3()
                }
                correlation_manager.remove_position(symbol)
                self._missing_position_counts.pop(symbol, None)
                del self.active_positions[symbol]
                continue
            
            for pos in positions:
                if pos["symbol"] == symbol:
                    self._missing_position_counts[symbol] = 0
                    if (
                        not pos_data.get("sl_order_id")
                        and not pos_data.get("tp1_order_id")
                        and float(pos_data.get("tp1", 0) or 0) > 0
                    ):
                        self._place_position_exit_orders(symbol, pos_data)
                        logger.info(f"{symbol} koruma emirleri yeniden yerleştirildi (TP/SL)")

                    if not pos_data.get("sl_order_id") and float(pos_data.get("sl_trigger", 0) or 0) > 0:
                        if self._replace_stop_limit(
                            symbol,
                            pos_data,
                            float(pos_data.get("sl_trigger", pos_data.get("stop_loss", 0)) or 0),
                            float(pos_data.get("sl_limit", pos_data.get("sl_trigger", 0)) or 0),
                        ):
                            logger.info(f"{symbol} eksik SL emri yeniden yerleştirildi")

                    current_price = self.scanner.get_current_price(symbol)
                    unrealised_pnl = pos["unrealised_pnl"]
                    pos_data["unrealised_pnl"] = unrealised_pnl
                    pos_data["size"] = pos["size"]

                    self._check_max_hold_time(symbol, pos_data, current_price)
                    self._update_trailing_stop(symbol, current_price)

                    initial_size = float(pos_data.get("original_size", 0) or 0)
                    live_size = float(pos.get("size", 0) or 0)
                    atr = float(pos_data.get("entry_atr", 0) or 0)
                    tp1_pct = float(pos_data.get("tp1_percent", STRATEGY_CONFIG.get("tp1_percent", 40)))
                    tp2_pct = float(pos_data.get("tp2_percent", STRATEGY_CONFIG.get("tp2_percent", 40)))
                    tp1_remaining_ratio = max(0.0, 1 - tp1_pct / 100)
                    tp2_remaining_ratio = max(0.0, 1 - (tp1_pct + tp2_pct) / 100)

                    if initial_size > 0 and live_size <= initial_size * (tp1_remaining_ratio + 0.02):
                        if not pos_data.get("tp1_hit"):
                            pos_data["tp1_hit"] = True
                        be_limit_offset = float(STRATEGY_CONFIG.get("sl_be_offset_atr", 0.15)) * atr
                        be_trigger = float(pos_data.get("entry_price", current_price))
                        tp1_qty = float(pos_data.get("tp1_qty", 0) or 0)
                        if tp1_qty <= 0:
                            tp1_qty = max(0.0, initial_size - live_size)
                        tp1_realized = float(pos_data.get("tp1_realized_pnl", 0) or 0)
                        if abs(tp1_realized) <= 1e-9 and tp1_qty > 0:
                            tp1_realized = self._calculate_leg_pnl(
                                str(pos_data.get("side", "Buy")),
                                float(pos_data.get("entry_price", 0) or 0),
                                float(pos_data.get("tp1", 0) or 0),
                                tp1_qty,
                            )
                            pos_data["tp1_realized_pnl"] = tp1_realized
                        cumulative_realized = tp1_realized + float(pos_data.get("tp2_realized_pnl", 0) or 0)
                        pos_data["realized_pnl"] = cumulative_realized
                        if pos_data.get("side") == "Buy":
                            be_limit = be_trigger - be_limit_offset
                        else:
                            be_limit = be_trigger + be_limit_offset
                        if (not pos_data.get("tp1_be_applied")) and self._replace_stop_limit(symbol, pos_data, be_trigger, be_limit):
                            pos_data["tp1_be_applied"] = True
                            if pos_data.get("trade_id"):
                                update_trade(
                                    trade_id=pos_data["trade_id"],
                                    tp1_hit=True,
                                    partial_closed_size=max(0.0, initial_size - live_size),
                                )
                            self._notify_telegram_sync(
                                f"🎯 <b>TP1 DOLDU</b>\n\n"
                                f"<b>Sembol:</b> {symbol}\n"
                                f"<b>Bu Kademe Kar:</b> {self._format_pnl(tp1_realized)}\n"
                                f"<b>Kümülatif Kar:</b> {self._format_pnl(cumulative_realized)}\n"
                                f"<b>Yeni SL Trigger:</b> BE (${be_trigger:.4f})"
                            )
                        elif pos_data.get("tp1_hit") and not pos_data.get("tp1_be_applied"):
                            logger.warning(f"{symbol} TP1 görüldü ancak BE SL henüz uygulanamadı, tekrar denenecek")

                    if initial_size > 0 and pos_data.get("tp1_hit") and not pos_data.get("tp2_hit") and live_size <= initial_size * (tp2_remaining_ratio + 0.02):
                        pos_data["tp2_hit"] = True
                        tp2_qty = float(pos_data.get("tp2_qty", 0) or 0)
                        if tp2_qty <= 0:
                            tp2_qty = max(0.0, initial_size * tp1_remaining_ratio - live_size)
                        tp2_realized = self._calculate_leg_pnl(
                            str(pos_data.get("side", "Buy")),
                            float(pos_data.get("entry_price", 0) or 0),
                            float(pos_data.get("tp2", 0) or 0),
                            tp2_qty,
                        )
                        pos_data["tp2_realized_pnl"] = tp2_realized
                        cumulative_realized = float(pos_data.get("tp1_realized_pnl", 0) or 0) + tp2_realized
                        pos_data["realized_pnl"] = cumulative_realized
                        tf = str(EXECUTION_CONFIG.get("timeframe", "5"))
                        sl_df = self.scanner.get_klines(symbol, interval=tf)
                        if sl_df is not None and len(sl_df) >= 30:
                            ema21 = float(sl_df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
                            atr_now = float((sl_df["high"] - sl_df["low"]).rolling(14).mean().iloc[-1] or atr)
                        else:
                            ema21 = float(pos_data.get("entry_price", current_price))
                            atr_now = atr

                        trigger_off = float(STRATEGY_CONFIG.get("sl_trail_trigger_offset_atr", 0.10)) * atr_now
                        limit_off = float(STRATEGY_CONFIG.get("sl_trail_limit_offset_atr", 0.15)) * atr_now
                        current_sl = float(pos_data.get("sl_trigger", pos_data.get("stop_loss", 0)) or 0)
                        if pos_data.get("side") == "Buy":
                            trail_trigger = ema21 - trigger_off
                            trail_limit = trail_trigger - limit_off
                            if trail_trigger <= current_sl:
                                continue
                        else:
                            trail_trigger = ema21 + trigger_off
                            trail_limit = trail_trigger + limit_off
                            if current_sl > 0 and trail_trigger >= current_sl:
                                continue
                        if self._replace_stop_limit(symbol, pos_data, trail_trigger, trail_limit):
                            self._notify_telegram_sync(
                                f"📈 <b>TP2 DOLDU</b>\n\n"
                                f"<b>Sembol:</b> {symbol}\n"
                                f"<b>Bu Kademe Kar:</b> {self._format_pnl(tp2_realized)}\n"
                                f"<b>Kümülatif Kar:</b> {self._format_pnl(cumulative_realized)}\n"
                                f"<b>Yeni SL Trigger:</b> ${float(pos_data.get('sl_trigger', trail_trigger)):.4f}"
                            )

                    if pos_data.get("tp2_hit"):
                        self._update_tp2_trailing_on_close(symbol, pos_data)
                    
                    logger.info(f"{symbol} {pos_data['side']} PnL: ${unrealised_pnl:.2f}")
                    break

    def run(self) -> None:
        logger.info("=" * 50)
        logger.info("🤖 CRYPTO BOT BAŞLATILIYOR")
        logger.info("=" * 50)
        logger.info(f"Zaman Dilimi: UTC+{TIMEZONE_OFFSET}")
        logger.info(f"Kaldıraç: {RISK_CONFIG['leverage']}x")
        logger.info(f"Max Günlük Kayıp: {RISK_CONFIG['daily_max_losses']}")
        logger.info(f"Trailing Stop: {'Aktif' if RISK_CONFIG.get('trailing_stop_enabled') else 'Pasif'}")
        logger.info(f"Tarama Aralığı: {SCAN_CONFIG['scan_interval_seconds']}s")
        logger.info(
            f"Min RR Filtresi: {max(float(STRATEGY_CONFIG.get('rr_hard_min', 1.0)), float(STRATEGY_CONFIG.get('min_rr', 1.35))):.2f}"
        )
        logger.info(f"Trading Modu: {ACTIVE_TRADING_MODE}")
        logger.info(
            "Risk Profili: "
            f"fixed_sl={bool(RISK_CONFIG.get('use_fixed_sl_usdt', False))}, "
            f"adaptive_sl={bool(RISK_CONFIG.get('adaptive_sl_enabled', False))}, "
            f"min_risk_usdt={float(RISK_CONFIG.get('min_risk_per_trade', 0) or 0):.2f}, "
            f"fixed_sl_usdt={float(RISK_CONFIG.get('fixed_sl_usdt', 0) or 0):.2f}, "
            f"min_notional={float(RISK_CONFIG.get('min_position_size_usdt', 0) or 0):.2f}, "
            f"max_notional={float(RISK_CONFIG.get('max_position_size_usdt', 0) or 0):.2f}"
        )
        logger.info(
            "Efektif Hedefler: "
            f"min_sl_usdt={float(RISK_CONFIG.get('min_effective_sl_usdt', 0) or 0):.2f}, "
            f"min_tp1_usdt={float(RISK_CONFIG.get('min_effective_tp1_usdt', 0) or 0):.2f}, "
            f"preferred_notional={float(RISK_CONFIG.get('min_preferred_notional_usdt', 0) or 0):.2f}, "
            f"enforce={bool(RISK_CONFIG.get('enforce_min_effective_targets', True))}"
        )
        logger.info("TF Seti: confirm=15m, trend=5m, entry=1m")
        logger.info(f"Sembol Sayısı: {len(self.scanner.symbols)}")
        logger.info(f"Trading Durumu: {'ACTIVE' if self.trading_enabled else 'PAUSED'}")
        logger.info("=" * 50)

        auth_ok, auth_msg = self.scanner.verify_private_auth()
        if not auth_ok:
            logger.error(f"Bybit private auth doğrulaması başarısız: {auth_msg}")
            self._notify_telegram_sync(
                "❌ <b>Auth doğrulama başarısız</b>\n"
                "Private endpoint erişimi yok. API key/secret ve izinleri kontrol edin."
            )
            self.running = False
            self.stop_event.set()
            return

        logger.info("Bybit private auth doğrulandı")
        self._bootstrap_existing_positions(notify=True)
        
        self._notify_telegram_sync(
            "🤖 <b>Bot başlatıldı!</b>\n\n"
            "Komutlar: /status, /balance, /history, /logs, /pause, /resume, /stop, /help"
        )

        while self.running:
            try:
                self._scan_and_trade()
                if not self.running:
                    break
                self._monitor_positions()
                if not self.running:
                    break
                
                now = time.time()
                if now - self._last_balance_update > 30:
                    try:
                        self._cached_balance = self.scanner.get_balance()
                        self._last_balance_update = now
                    except:
                        pass
                
                if self.stop_event.wait(SCAN_CONFIG["scan_interval_seconds"]):
                    break
                
            except KeyboardInterrupt:
                logger.info("Bot durduruluyor...")
                self.running = False
                self.stop_event.set()
                self._notify_shutdown_once("KeyboardInterrupt")
            except Exception as e:
                logger.error(f"Ana döngü hatası: {e}")
                self._notify_telegram_sync(f"Ana döngü hatası: {e}")
                
                self.restart_count += 1
                if self.restart_count <= self.max_restarts:
                    wait_time = min(60, 5 * self.restart_count)
                    logger.info(f"Yeniden başlatılıyor... ({self.restart_count}/{self.max_restarts}) - {wait_time}s bekleniyor")
                    time.sleep(wait_time)
                else:
                    logger.error("Max restart sayısına ulaşıldı, bot durduruluyor")
                    self.running = False

        self._notify_shutdown_once("Ana döngü sonlandı")
        self._release_instance_lock()


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
