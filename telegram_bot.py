import asyncio
import threading
import time
from typing import Optional, Callable
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from logger import logger


class TelegramBot:
    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = str(TELEGRAM_CHAT_ID)
        self.enabled = bool(self.token and self.chat_id)
        self.bot: Optional[Bot] = None
        self.app: Optional[Application] = None
        self.callbacks = {}
        self._poll_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        
        if self.enabled:
            self.bot = Bot(token=self.token)

    def on(self, name: str, func: Callable):
        self.callbacks[name] = func

    def setup(self):
        if not self.enabled:
            return
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("status", self._status))
        self.app.add_handler(CommandHandler("balance", self._balance))
        self.app.add_handler(CommandHandler("history", self._history))
        self.app.add_handler(CommandHandler("logs", self._logs))
        self.app.add_handler(CommandHandler("pause", self._pause))
        self.app.add_handler(CommandHandler("resume", self._resume))
        self.app.add_handler(CommandHandler("stop", self._stop))
        self.app.add_handler(CommandHandler("help", self._help))
        self.app.add_handler(CommandHandler("start", self._resume))

    def start(self):
        if not self.enabled or not self.app:
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._poll_loop = loop
        loop.run_until_complete(self._poll())

    async def _poll(self):
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        self._ready.set()
        while True:
            await asyncio.sleep(1)

    def _reply(self, update: Update, text: str):
        asyncio.create_task(update.message.reply_text(text, parse_mode=ParseMode.HTML))

    async def _status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("status")
        if cb:
            data = cb()
            pos = "Açık" if data.get("has_open_position") else "Kapalı"
            can = "✅" if data.get("can_trade") else "❌"
            trading_mode = "▶️ ACTIVE" if data.get("trading_enabled", True) else "⏸️ PAUSED"
            extra = ""
            if data.get("current_symbol"):
                p = data.get("position_data", {})
                extra = f"\n<b>Pozisyon:</b> {data['current_symbol']} {p.get('side','')} ${p.get('unrealised_pnl',0):.2f}"
            await update.message.reply_text(
                f"🤖 <b>DURUM</b>\n\n"
                f"<b>Mod:</b> {trading_mode}\n"
                f"<b>Pozisyon:</b> {pos}{extra}\n"
                f"<b>Kayıp:</b> {data.get('daily_losses',0)}/{data.get('max_daily_losses',0)}\n"
                f"<b>Trade:</b> {data.get('daily_trades',0)}/{data.get('max_daily_trades',0)}\n"
                f"<b>Günlük PnL:</b> ${data.get('daily_pnl',0):.2f} ({data.get('daily_pnl_percent',0):.2f}%)\n"
                f"<b>İşlem:</b> {can}",
                parse_mode=ParseMode.HTML
            )

    async def _pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("pause")
        if cb:
            cb()
            await update.message.reply_text("⏸️ Trading pause aktif")

    async def _resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("resume")
        if cb:
            cb()
            await update.message.reply_text("▶️ Trading resume aktif")

    async def _balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("balance")
        if cb:
            data = cb()
            await update.message.reply_text(
                f"💰 <b>BAKİYE</b>\n\n<b>Toplam:</b> ${data.get('total',0):.2f}\n<b>Kullanılabilir:</b> ${data.get('available',0):.2f}",
                parse_mode=ParseMode.HTML
            )

    async def _history(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("history")
        if cb:
            trades = cb()
            if not trades:
                await update.message.reply_text("📭 Boş")
            else:
                lines = ["📋 <b>İŞLEMLER</b>"]
                for t in trades[:5]:
                    p = float(t.get("pnl") or 0)
                    e = "✅" if p >= 0 else "❌"
                    status = t.get("status", "-")
                    side = t.get("side", "-")
                    lines.append(f"{e} {t.get('symbol','-')} {side} [{status}] | ${p:.2f}")
                await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("stop")
        if cb:
            cb()
            await update.message.reply_text("🛑 Durduruluyor...")

    async def _logs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        cb = self.callbacks.get("logs")
        if cb:
            text = cb()
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def _help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != self.chat_id:
            return
        await update.message.reply_text(
            "🤖 <b>KOMUTLAR</b>\n\n/status /balance /history /logs /pause /resume /stop /help",
            parse_mode=ParseMode.HTML
        )

    async def send(self, text: str) -> bool:
        if not self.enabled or not self.bot:
            return False
        try:
            await asyncio.wait_for(
                self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML),
                timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            logger.error("[TG] send error: Timed out")
            return False
        except Exception as e:
            logger.error(f"[TG] send error: {e}")
            return False

    def send_now(self, text: str) -> bool:
        if not self.enabled:
            logger.warning("[TG] send_now skipped: telegram disabled")
            return False

        if self._ready.wait(timeout=2.0) and self._poll_loop and self._poll_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self.send(text), self._poll_loop)
                result = future.result(timeout=3)
                if result:
                    return True
                logger.error("[TG] send_now failed on poll loop")
            except Exception as e:
                logger.error(f"[TG] send_now scheduling error: {e}")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.send(text))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
            if result:
                return True
            logger.error("[TG] send_now fallback send failed")
        except Exception as e:
            logger.error(f"[TG] send_now fallback error: {e}")
        
        return False

    def signal_now(self, sig: str, sym: str, entry: float, sl: float, tp1: float, tp2: float, rr: float) -> bool:
        e = "🟢" if sig == "LONG" else "🔴"
        def fmt(v: float) -> str:
            x = abs(float(v))
            if x >= 1000:
                return f"{v:.2f}"
            if x >= 1:
                return f"{v:.4f}"
            if x >= 0.01:
                return f"{v:.6f}"
            return f"{v:.8f}"
        return self.send_now(
            f"{e} <b>{sig}</b>\n{sym} @ ${fmt(entry)}\nSL: ${fmt(sl)}\nTP1: ${fmt(tp1)}\nTP2: ${fmt(tp2)}\nR:R 1:{rr:.1f}"
        )

    def closed_now(
        self,
        sym: str,
        side: str,
        pnl: float,
        reason: str = "",
        leg_pnl: Optional[float] = None,
        cumulative_pnl: Optional[float] = None,
        balance_before: Optional[float] = None,
        balance_after: Optional[float] = None,
    ) -> bool:
        e = "✅" if pnl >= 0 else "❌"
        p = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        leg_line = ""
        cumulative_line = ""
        balance_line = ""
        if leg_pnl is not None:
            leg_line = f"\nSon Kademe: {'+' if leg_pnl >= 0 else '-'}${abs(leg_pnl):.2f}"
        if cumulative_pnl is not None:
            cumulative_line = f"\nKümülatif: {'+' if cumulative_pnl >= 0 else '-'}${abs(cumulative_pnl):.2f}"
        if balance_before is not None and balance_after is not None:
            balance_line = f"\nBakiye: ${balance_before:.2f} -> ${balance_after:.2f}"
        reason_line = f"\nNeden: {reason}" if reason else ""
        return self.send_now(
            f"{e} <b>KAPANDI</b>\n{sym} {side}\nNet işlem sonucu: {p}{balance_line}{leg_line}{cumulative_line}{reason_line}"
        )

    async def signal(self, sig: str, sym: str, entry: float, sl: float, tp1: float, tp2: float, rr: float):
        e = "🟢" if sig == "LONG" else "🔴"
        def fmt(v: float) -> str:
            x = abs(float(v))
            if x >= 1000:
                return f"{v:.2f}"
            if x >= 1:
                return f"{v:.4f}"
            if x >= 0.01:
                return f"{v:.6f}"
            return f"{v:.8f}"
        await self.send(f"{e} <b>{sig}</b>\n{sym} @ ${fmt(entry)}\nSL: ${fmt(sl)}\nTP1: ${fmt(tp1)}\nTP2: ${fmt(tp2)}\nR:R 1:{rr:.1f}")

    async def closed(
        self,
        sym: str,
        side: str,
        pnl: float,
        reason: str = "",
        leg_pnl: Optional[float] = None,
        cumulative_pnl: Optional[float] = None,
        balance_before: Optional[float] = None,
        balance_after: Optional[float] = None,
    ):
        e = "✅" if pnl >= 0 else "❌"
        p = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        leg_line = ""
        cumulative_line = ""
        balance_line = ""
        if leg_pnl is not None:
            leg_line = f"\nSon Kademe: {'+' if leg_pnl >= 0 else '-'}${abs(leg_pnl):.2f}"
        if cumulative_pnl is not None:
            cumulative_line = f"\nKümülatif: {'+' if cumulative_pnl >= 0 else '-'}${abs(cumulative_pnl):.2f}"
        if balance_before is not None and balance_after is not None:
            balance_line = f"\nBakiye: ${balance_before:.2f} -> ${balance_after:.2f}"
        reason_line = f"\nNeden: {reason}" if reason else ""
        await self.send(
            f"{e} <b>KAPANDI</b>\n{sym} {side}\nNet işlem sonucu: {p}{balance_line}{leg_line}{cumulative_line}{reason_line}"
        )


tg = TelegramBot()
