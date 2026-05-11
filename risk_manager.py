from datetime import datetime, timedelta
from typing import Optional
from config import RISK_CONFIG


class RiskManager:
    def __init__(self):
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.daily_loss_r = 0.0
        self.direction_losses = {"LONG": 0, "SHORT": 0}
        self.last_loss_date: Optional[datetime] = None
        self.has_open_position = False
        self.current_symbol: Optional[str] = None
        self.current_signal: Optional[str] = None
        self.current_risk_amount: float = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.day_start_balance: Optional[float] = None
        self._last_reset_date: Optional[str] = None

    def reset_daily_if_needed(self, now: datetime, current_balance: float = 0.0) -> None:
        today = now.date().isoformat()
        if self._last_reset_date != today:
            self.daily_losses = 0
            self.consecutive_losses = 0
            self.daily_loss_r = 0.0
            self.direction_losses = {"LONG": 0, "SHORT": 0}
            self.daily_pnl = 0.0
            self.daily_trades = 0
            if current_balance > 0:
                self.day_start_balance = current_balance
            self._last_reset_date = today

    def set_day_start_balance(self, balance: float, now: datetime) -> None:
        self.reset_daily_if_needed(now, balance)
        if self.day_start_balance is None and balance > 0:
            self.day_start_balance = balance

    def get_daily_pnl_percent(self) -> float:
        if not self.day_start_balance or self.day_start_balance <= 0:
            return 0.0
        return (self.daily_pnl / self.day_start_balance) * 100

    def can_open_trade(self, now: datetime, current_balance: float = 0.0) -> tuple[bool, str]:
        self.reset_daily_if_needed(now, current_balance)

        if self.has_open_position:
            return False, "Açık pozisyon var - yeni işlem açılamaz"

        if self.daily_losses >= RISK_CONFIG["daily_max_losses"]:
            return False, f"Günlük max kayıp ({RISK_CONFIG['daily_max_losses']}) aşıldı - yarın tekrar dene"

        if self.daily_loss_r >= float(RISK_CONFIG.get("max_daily_loss_r", 3.0)):
            return False, f"Günlük max loss R limiti aşıldı ({self.daily_loss_r:.2f}R)"

        if self.daily_trades >= RISK_CONFIG.get("max_daily_trades", 3):
            return False, f"Günlük max işlem ({RISK_CONFIG.get('max_daily_trades', 3)}) sınırına ulaşıldı"

        pnl_pct = self.get_daily_pnl_percent()
        if pnl_pct >= RISK_CONFIG.get("daily_profit_target_percent", 3.0):
            return False, f"Günlük kar hedefi gerçekleşti ({pnl_pct:.2f}%)"

        if pnl_pct <= -abs(RISK_CONFIG.get("daily_loss_limit_percent", 2.0)):
            return False, f"Günlük zarar limiti aşıldı ({pnl_pct:.2f}%)"

        cooldown_minutes = int(RISK_CONFIG.get("loss_cooldown_minutes", 180))
        if self.last_loss_date and cooldown_minutes > 0:
            elapsed = now - self.last_loss_date
            cooldown_delta = timedelta(minutes=cooldown_minutes)
            if elapsed < cooldown_delta:
                remain = int((cooldown_delta - elapsed).total_seconds() // 60)
                return False, f"Stop sonrası cooldown aktif ({remain} dk kaldı)"

        return True, "İşlem açılabilir"

    def can_open_direction(self, signal: str) -> tuple[bool, str]:
        sig = str(signal or "").upper()
        if sig not in ("LONG", "SHORT"):
            return False, "Yön bilinmiyor"
        max_same = int(RISK_CONFIG.get("max_same_direction_losses", 2))
        if self.direction_losses.get(sig, 0) >= max_same:
            return False, f"{sig} yönünde maksimum deneme aşıldı ({max_same})"
        return True, "Yön uygun"

    def register_loss(self, now: datetime) -> None:
        self.reset_daily_if_needed(now)
        self.daily_losses += 1
        self.consecutive_losses += 1
        self.last_loss_date = now
        self.has_open_position = False
        self.current_symbol = None
        self.current_signal = None
        self.current_risk_amount = 0.0

    def register_win(self) -> None:
        self.consecutive_losses = 0
        self.has_open_position = False
        self.current_symbol = None
        self.current_signal = None
        self.current_risk_amount = 0.0

    def open_position(self, symbol: str, signal: Optional[str] = None, risk_amount: float = 0.0) -> None:
        self.has_open_position = True
        self.current_symbol = symbol
        self.current_signal = (signal or "").upper() if signal else None
        self.current_risk_amount = max(0.0, float(risk_amount or 0.0))
        self.daily_trades += 1

    def close_position(
        self,
        was_profit: bool,
        pnl: float,
        now: datetime,
        signal: Optional[str] = None,
        risk_amount: float = 0.0,
    ) -> None:
        self.reset_daily_if_needed(now)
        self.daily_pnl += pnl
        sig = (signal or self.current_signal or "").upper()
        effective_risk = float(risk_amount or self.current_risk_amount or 0.0)

        if was_profit:
            if sig in self.direction_losses:
                self.direction_losses[sig] = 0
            self.register_win()
        else:
            if effective_risk > 0:
                self.daily_loss_r += abs(float(pnl)) / effective_risk
            if sig in self.direction_losses:
                self.direction_losses[sig] = self.direction_losses.get(sig, 0) + 1
            self.register_loss(now)

    def should_stop_for_day(self, now: datetime, current_balance: float = 0.0) -> tuple[bool, str]:
        self.reset_daily_if_needed(now, current_balance)
        pnl_pct = self.get_daily_pnl_percent()
        profit_target = RISK_CONFIG.get("daily_profit_target_percent", 3.0)
        loss_limit = abs(RISK_CONFIG.get("daily_loss_limit_percent", 2.0))

        if pnl_pct >= profit_target:
            return True, f"Günlük kar hedefi ({profit_target:.2f}%) gerçekleşti: {pnl_pct:.2f}%"

        if pnl_pct <= -loss_limit:
            return True, f"Günlük zarar limiti (-{loss_limit:.2f}%) aşıldı: {pnl_pct:.2f}%"

        max_loss_r = float(RISK_CONFIG.get("max_daily_loss_r", 3.0))
        if self.daily_loss_r >= max_loss_r:
            return True, f"Günlük max loss R limiti ({max_loss_r:.2f}R) aşıldı: {self.daily_loss_r:.2f}R"

        if self.daily_trades >= RISK_CONFIG.get("max_daily_trades", 3) and not self.has_open_position:
            return True, f"Günlük max işlem sayısı ({self.daily_trades}) tamamlandı"

        return False, "Devam"

    def calculate_position_size(
        self, entry: float, stop_loss: float, balance: float, risk_multiplier: float = 1.0
    ) -> float:
        size_mode = str(RISK_CONFIG.get("position_size_mode", "risk_based")).lower()
        max_notional = float(RISK_CONFIG.get("max_position_size_usdt", 100))

        if size_mode == "fixed_notional":
            if entry <= 0 or max_notional <= 0:
                return 0
            return round(max_notional / entry, 3)

        risk_mode = str(RISK_CONFIG.get("risk_mode", "hybrid")).lower()
        fixed_risk = float(RISK_CONFIG.get("max_risk_per_trade", 15.0))
        percent_risk = max(0.0, float(RISK_CONFIG.get("risk_percent_per_trade", 1.0)))
        balance_risk = balance * (percent_risk / 100)

        if risk_mode == "fixed":
            risk_amount = fixed_risk
        elif risk_mode == "percent":
            risk_amount = balance_risk
        else:
            risk_amount = min(fixed_risk, balance_risk) if balance > 0 else fixed_risk

        risk_amount = risk_amount * risk_multiplier

        price_diff = abs(entry - stop_loss)

        if price_diff == 0 or risk_amount <= 0:
            return 0

        position_size = risk_amount / price_diff

        notional = position_size * entry

        if notional > max_notional:
            position_size = max_notional / entry

        return round(position_size, 3)

    def get_status(self) -> dict:
        daily_pnl_pct = self.get_daily_pnl_percent()
        max_losses = RISK_CONFIG["daily_max_losses"]
        max_trades = RISK_CONFIG.get("max_daily_trades", 3)
        profit_target = RISK_CONFIG.get("daily_profit_target_percent", 3.0)
        loss_limit = abs(RISK_CONFIG.get("daily_loss_limit_percent", 2.0))
        can_trade = (
            self.daily_losses < max_losses
            and self.daily_loss_r < float(RISK_CONFIG.get("max_daily_loss_r", 3.0))
            and self.daily_trades < max_trades
            and daily_pnl_pct < profit_target
            and daily_pnl_pct > -loss_limit
            and not self.has_open_position
        )
        return {
            "daily_losses": self.daily_losses,
            "consecutive_losses": self.consecutive_losses,
            "max_daily_losses": max_losses,
            "daily_trades": self.daily_trades,
            "max_daily_trades": max_trades,
            "daily_loss_r": self.daily_loss_r,
            "max_daily_loss_r": float(RISK_CONFIG.get("max_daily_loss_r", 3.0)),
            "long_losses": self.direction_losses.get("LONG", 0),
            "short_losses": self.direction_losses.get("SHORT", 0),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": daily_pnl_pct,
            "has_open_position": self.has_open_position,
            "current_symbol": self.current_symbol,
            "can_trade": can_trade,
        }
