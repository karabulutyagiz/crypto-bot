import pandas as pd
import math
from typing import List, Optional
import time
import random
import threading
from pybit.unified_trading import HTTP
from logger import logger
from config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_TESTNET,
    BYBIT_DEMO_TRADING,
    SCAN_CONFIG,
    TIMEZONE_OFFSET,
)
from datetime import datetime, timedelta


class Scanner:
    def __init__(self):
        self.session = HTTP(
            testnet=BYBIT_TESTNET,
            demo=BYBIT_DEMO_TRADING,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
        )
        self.symbols: List[str] = []
        self._symbol_meta: dict[str, dict] = {}
        self._kline_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}
        self._ticker_cache: tuple[float, dict[str, dict]] = (0.0, {})
        self._kline_lock = threading.Lock()
        self._last_kline_request_ts = 0.0
        self._kline_rate_limit_until = 0.0
        self._load_symbols()

    @staticmethod
    def _clone_kline_df(
        df: pd.DataFrame,
        *,
        fetched_at: float,
        interval: str,
        source: str,
        stale: bool,
    ) -> pd.DataFrame:
        out = df.copy()
        out.attrs["kline_fetched_at"] = float(fetched_at)
        out.attrs["kline_interval"] = str(interval)
        out.attrs["kline_source"] = str(source)
        out.attrs["kline_stale"] = bool(stale)
        out.attrs["kline_age_seconds"] = max(0.0, time.time() - float(fetched_at))
        return out

    @staticmethod
    def is_stale_kline(df: Optional[pd.DataFrame]) -> bool:
        return bool(getattr(df, "attrs", {}).get("kline_stale", False)) if df is not None else True

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_symbols(self) -> None:
        try:
            response = self.session.get_instruments_info(category="linear")
            min_turnover = float(SCAN_CONFIG.get("min_turnover_usdt", 2000000))
            max_symbols = int(SCAN_CONFIG.get("max_symbols", 25))
            scan_all_perps = bool(SCAN_CONFIG.get("scan_all_linear_perps", True))

            usdt_perp_symbols = {
                item["symbol"]
                for item in response["result"]["list"]
                if item.get("quoteCoin") == "USDT" and item.get("contractType") == "LinearPerpetual"
            }

            if scan_all_perps:
                all_symbols = sorted(usdt_perp_symbols)
                if max_symbols > 0 and len(all_symbols) > max_symbols:
                    ranked: list[tuple[str, float]] = []
                    try:
                        tickers = self.session.get_tickers(category="linear")
                        for item in tickers.get("result", {}).get("list", []):
                            symbol = item.get("symbol")
                            if symbol not in usdt_perp_symbols:
                                continue
                            turnover = self._to_float(item.get("turnover24h", item.get("volume24h", 0)))
                            ranked.append((symbol, turnover))
                    except Exception:
                        ranked = []

                    if ranked:
                        ranked.sort(key=lambda x: x[1], reverse=True)
                        self.symbols = [sym for sym, _ in ranked[:max_symbols]]
                    else:
                        self.symbols = all_symbols[:max_symbols]
                else:
                    self.symbols = all_symbols
                if not self.symbols:
                    raise RuntimeError("USDT linear perpetual sembol bulunamadı")
                logger.info(
                    f"[Scanner] USDT perpetual evreni yüklendi: {len(self.symbols)} sembol "
                    f"(scan_all={scan_all_perps}, max_symbols={max_symbols})"
                )
                return

            candidates = []
            try:
                tickers = self.session.get_tickers(category="linear")
                for item in tickers.get("result", {}).get("list", []):
                    symbol = item.get("symbol")
                    if symbol not in usdt_perp_symbols:
                        continue
                    turnover = self._to_float(item.get("turnover24h", item.get("volume24h", 0)))
                    if turnover < min_turnover:
                        continue
                    candidates.append((symbol, turnover))
            except Exception:
                candidates = []

            if not candidates:
                for item in response["result"]["list"]:
                    if item.get("quoteCoin") != "USDT":
                        continue
                    if item.get("contractType") != "LinearPerpetual":
                        continue

                    turnover = self._to_float(item.get("turnover24h", item.get("volume24h", 0)))
                    if turnover < min_turnover:
                        continue

                    candidates.append((item["symbol"], turnover))

            candidates.sort(key=lambda x: x[1], reverse=True)
            self.symbols = [sym for sym, _ in candidates[:max_symbols]]
            if not self.symbols:
                self.symbols = [
                    "BTCUSDT",
                    "ETHUSDT",
                    "SOLUSDT",
                    "XRPUSDT",
                    "DOGEUSDT",
                    "ADAUSDT",
                    "AVAXUSDT",
                    "LINKUSDT",
                    "LTCUSDT",
                    "BNBUSDT",
                ]
            logger.info(f"[Scanner] {len(self.symbols)} USDT perpetual yüklendi: {', '.join(self.symbols)}")
        except Exception as e:
            logger.error(f"[Scanner] Sembol yükleme hatası: {e}")
            self.symbols = [
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "XRPUSDT",
                "DOGEUSDT",
                "ADAUSDT",
                "AVAXUSDT",
                "LINKUSDT",
                "LTCUSDT",
                "BNBUSDT",
            ]

    def get_klines(self, symbol: str, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        tf = str(interval or SCAN_CONFIG["timeframe"])
        limit = int(SCAN_CONFIG["kline_limit"])
        ttl = int(SCAN_CONFIG.get("kline_cache_ttl_seconds", 12))
        stale_ttl = max(ttl, int(SCAN_CONFIG.get("kline_stale_ttl_seconds", 180)))
        min_request_interval = max(0.0, float(SCAN_CONFIG.get("kline_min_request_interval_seconds", 0.12)))
        max_retries = max(0, int(SCAN_CONFIG.get("kline_max_retries", 4)))
        retry_base = max(0.1, float(SCAN_CONFIG.get("kline_retry_base_seconds", 0.6)))
        retry_max = max(retry_base, float(SCAN_CONFIG.get("kline_retry_max_seconds", 8.0)))
        cache_key = (symbol, tf, limit)

        cached = self._kline_cache.get(cache_key)
        now_ts = time.time()
        if cached:
            ts, df_cached = cached
            if now_ts - ts <= ttl:
                return self._clone_kline_df(df_cached, fetched_at=ts, interval=tf, source="cache_fresh", stale=False)

        if now_ts < self._kline_rate_limit_until:
            if cached:
                ts, df_cached = cached
                if now_ts - ts <= stale_ttl:
                    logger.debug(f"[Scanner] {symbol} stale kline cache kullanildi (global rate-limit cooldown)")
                    return self._clone_kline_df(df_cached, fetched_at=ts, interval=tf, source="cache_stale", stale=True)
            logger.debug(
                f"[Scanner] {symbol} kline atlandi (global cooldown aktif, "
                f"kalan={self._kline_rate_limit_until - now_ts:.2f}s)"
            )
            return None

        for attempt in range(max_retries + 1):
            try:
                with self._kline_lock:
                    now_ts = time.time()
                    if now_ts < self._kline_rate_limit_until:
                        if cached:
                            ts, df_cached = cached
                            if now_ts - ts <= stale_ttl:
                                logger.debug(f"[Scanner] {symbol} stale kline cache kullanildi (cooldown icinde)")
                                return self._clone_kline_df(df_cached, fetched_at=ts, interval=tf, source="cache_stale", stale=True)
                        logger.debug(
                            f"[Scanner] {symbol} kline atlandi (cooldown icinde, "
                            f"kalan={self._kline_rate_limit_until - now_ts:.2f}s)"
                        )
                        return None

                    elapsed = time.time() - self._last_kline_request_ts
                    if elapsed < min_request_interval:
                        time.sleep(min_request_interval - elapsed)

                    response = self.session.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=tf,
                        limit=limit,
                    )
                    self._last_kline_request_ts = time.time()

                data = response["result"]["list"]
                if not data:
                    return None

                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df = df.iloc[::-1].reset_index(drop=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)

                fetched_at = time.time()
                self._kline_cache[cache_key] = (fetched_at, df)
                return self._clone_kline_df(df, fetched_at=fetched_at, interval=tf, source="api", stale=False)
            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = (
                    "10006" in msg
                    or "rate limit" in msg
                    or "too many visits" in msg
                    or "x-bapi-limit-reset-timestamp" in msg
                )

                if (not is_rate_limit) or attempt >= max_retries:
                    logger.error(f"[Scanner] {symbol} kline hatası: {e}")
                    if cached:
                        ts, stale_df = cached
                        logger.warning(f"[Scanner] {symbol} stale kline cache kullanıldı")
                        return self._clone_kline_df(stale_df, fetched_at=ts, interval=tf, source="cache_stale", stale=True)
                    return None

                backoff = min(retry_max, retry_base * (2 ** attempt))
                rate_limit_penalty = max(0.5, float(SCAN_CONFIG.get("kline_rate_limit_penalty_seconds", 1.2)))
                if "x-bapi-limit-reset-timestamp" in msg:
                    backoff = max(backoff, rate_limit_penalty)
                jitter = random.uniform(0.0, min(0.25, backoff * 0.2))
                sleep_for = backoff + jitter
                self._kline_rate_limit_until = max(self._kline_rate_limit_until, time.time() + sleep_for)
                if cached:
                    ts, stale_df = cached
                    if time.time() - ts <= stale_ttl:
                        logger.warning(
                            f"[Scanner] {symbol} stale kline cache kullanildi "
                            f"(rate-limit, bekleme={sleep_for:.2f}s)"
                        )
                        return self._clone_kline_df(stale_df, fetched_at=ts, interval=tf, source="cache_stale", stale=True)
                logger.warning(
                    f"[Scanner] {symbol} kline rate-limit/backoff "
                    f"(deneme {attempt + 1}/{max_retries + 1}, bekleme={sleep_for:.2f}s): {e}"
                )
                if not cached:
                    logger.warning(f"[Scanner] {symbol} cache yok, bu tur kline istegi atlandi")
                    return None
                time.sleep(sleep_for)

        if cached:
            ts, stale_df = cached
            logger.warning(f"[Scanner] {symbol} stale kline cache kullanıldı")
            return self._clone_kline_df(stale_df, fetched_at=ts, interval=tf, source="cache_stale", stale=True)
        return None

    def get_current_time_utc3(self) -> tuple:
        now_utc = datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)
        return (
            now_utc.hour,
            now_utc.minute,
            now_utc.day,
            now_utc.weekday(),
        )

    def get_balance(self) -> dict:
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            result = response["result"]["list"][0]
            
            total_balance = 0
            available_balance = -1.0
            unrealized_pnl = 0
            
            for coin in result.get("coin", []):
                if coin["coin"] == "USDT":
                    total_balance = self._to_float(coin.get("walletBalance", 0))
                    available_balance = self._to_float(coin.get("availableToWithdraw"), -1.0)
                    unrealized_pnl = self._to_float(coin.get("unrealisedPnl", 0))

                    if available_balance < 0:
                        available_balance = self._to_float(result.get("totalAvailableBalance"), -1.0)

                    if available_balance < 0:
                        margin_balance = self._to_float(result.get("totalMarginBalance"), -1.0)
                        initial_margin = self._to_float(result.get("totalInitialMargin"), -1.0)
                        if margin_balance >= 0 and initial_margin >= 0:
                            available_balance = max(0.0, margin_balance - initial_margin)

                    if available_balance < 0:
                        order_im = self._to_float(coin.get("totalOrderIM"), 0.0)
                        pos_im = self._to_float(coin.get("totalPositionIM"), 0.0)
                        available_balance = max(0.0, total_balance - order_im - pos_im)

                    break

            if available_balance < 0:
                available_balance = 0.0
            
            return {
                "total": total_balance,
                "available": available_balance,
                "unrealized_pnl": unrealized_pnl
            }
        except Exception as e:
            logger.error(f"[Scanner] Bakiye hatası: {e}")
            return {"total": 0, "available": 0, "unrealized_pnl": 0}

    def get_open_positions(self) -> List[dict]:
        try:
            response = self.session.get_positions(category="linear", settleCoin="USDT")
            positions = []
            for pos in response["result"]["list"]:
                size = self._to_float(pos.get("size", 0))
                if size > 0:
                    positions.append({
                        "symbol": pos["symbol"],
                        "side": pos["side"],
                        "size": size,
                        "entry_price": self._to_float(pos.get("avgPrice", 0)),
                        "unrealised_pnl": self._to_float(pos.get("unrealisedPnl", 0)),
                        "stop_loss": self._to_float(pos.get("stopLoss", 0)),
                        "take_profit": self._to_float(pos.get("takeProfit", 0)),
                    })
            return positions
        except Exception as e:
            logger.error(f"[Scanner] Pozisyon hatası: {e}")
            return []

    def has_open_position(self) -> bool:
        positions = self.get_open_positions()
        return len(positions) > 0

    def get_current_price(self, symbol: str) -> float:
        ttl_seconds = max(1.0, float(SCAN_CONFIG.get("ticker_snapshot_ttl_seconds", 3.0)))
        snapshots = self.get_ticker_snapshots(ttl_seconds=ttl_seconds)
        cached_item = snapshots.get(symbol, {}) if isinstance(snapshots, dict) else {}
        cached_last = self._to_float(cached_item.get("last", 0))
        if cached_last > 0:
            return cached_last

        try:
            response = self.session.get_tickers(category="linear", symbol=symbol)
            if response["retCode"] == 0 and response["result"]["list"]:
                item = response["result"]["list"][0]
                last_price = self._to_float(item.get("lastPrice", 0))
                if last_price > 0:
                    now = time.time()
                    last_ts, cached = self._ticker_cache
                    merged = dict(cached) if isinstance(cached, dict) else {}
                    merged[symbol] = {
                        "bid": self._to_float(item.get("bid1Price", 0)),
                        "ask": self._to_float(item.get("ask1Price", 0)),
                        "last": last_price,
                        "turnover24h": self._to_float(item.get("turnover24h", 0)),
                        "volume24h": self._to_float(item.get("volume24h", 0)),
                    }
                    self._ticker_cache = (now, merged)
                return last_price
            return 0
        except Exception as e:
            logger.error(f"[Scanner] Güncel fiyat hatası: {e}")
            return 0

    def get_ticker_snapshots(self, ttl_seconds: float = 8.0) -> dict[str, dict]:
        now = time.time()
        last_ts, cached = self._ticker_cache
        if cached and (now - last_ts) <= max(1.0, ttl_seconds):
            return cached

        try:
            response = self.session.get_tickers(category="linear")
            items = response.get("result", {}).get("list", []) if isinstance(response, dict) else []
            snapshots: dict[str, dict] = {}
            for item in items:
                symbol = str(item.get("symbol", "") or "").strip()
                if not symbol:
                    continue
                bid = self._to_float(item.get("bid1Price", 0))
                ask = self._to_float(item.get("ask1Price", 0))
                last = self._to_float(item.get("lastPrice", 0))
                turnover = self._to_float(item.get("turnover24h", 0))
                volume = self._to_float(item.get("volume24h", 0))
                snapshots[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "turnover24h": turnover,
                    "volume24h": volume,
                }

            self._ticker_cache = (now, snapshots)
            return snapshots
        except Exception as e:
            logger.error(f"[Scanner] Ticker snapshot hatası: {e}")
            return cached if cached else {}

    def is_kline_rate_limited(self) -> bool:
        return time.time() < self._kline_rate_limit_until

    def get_kline_rate_limit_remaining_seconds(self) -> float:
        return max(0.0, self._kline_rate_limit_until - time.time())

    def get_symbol_liquidity(self, symbol: str, snapshots: Optional[dict[str, dict]] = None) -> dict:
        snaps = snapshots if snapshots is not None else self.get_ticker_snapshots()
        item = snaps.get(symbol, {})
        bid = self._to_float(item.get("bid", 0))
        ask = self._to_float(item.get("ask", 0))
        last = self._to_float(item.get("last", 0))
        turnover = self._to_float(item.get("turnover24h", 0))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        spread_bps = ((ask - bid) / mid * 10000) if (bid > 0 and ask > 0 and mid > 0) else 0.0
        return {
            "turnover24h": turnover,
            "spread_bps": max(0.0, spread_bps),
            "has_book": bool(bid > 0 and ask > 0),
            "last_price": last,
        }

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if "110043" in msg or "leverage not modified" in msg:
                logger.debug(f"[Scanner] {symbol} kaldıraç zaten {leverage}x, değişiklik gerekmiyor")
                return True
            logger.error(f"[Scanner] Kaldıraç ayarlama hatası: {e}")
            return False

    def _get_symbol_meta(self, symbol: str) -> dict:
        if symbol in self._symbol_meta:
            return self._symbol_meta[symbol]

        try:
            response = self.session.get_instruments_info(category="linear", symbol=symbol)
            items = response["result"]["list"]
            if items:
                self._symbol_meta[symbol] = items[0]
                return items[0]
        except Exception as e:
            logger.error(f"[Scanner] Sembol metadata hatası ({symbol}): {e}")

        self._symbol_meta[symbol] = {}
        return {}

    def verify_private_auth(self) -> tuple[bool, str]:
        try:
            bal = self.session.get_wallet_balance(accountType="UNIFIED")
            if bal.get("retCode") != 0:
                return False, f"wallet-balance failed: {bal.get('retMsg', 'unknown error')}"

            pos = self.session.get_positions(category="linear", settleCoin="USDT")
            if pos.get("retCode") != 0:
                return False, f"position-list failed: {pos.get('retMsg', 'unknown error')}"

            return True, "OK"
        except Exception as e:
            return False, str(e)

    def normalize_qty(self, symbol: str, qty: float) -> float:
        meta = self._get_symbol_meta(symbol)
        lot = meta.get("lotSizeFilter", {}) if meta else {}

        min_qty = float(lot.get("minOrderQty", 0) or 0)
        max_qty = float(lot.get("maxOrderQty", 0) or 0)
        step = float(lot.get("qtyStep", 0) or 0)

        normalized = max(0.0, qty)

        if step > 0:
            normalized = int(normalized / step) * step

        if max_qty > 0:
            normalized = min(normalized, max_qty)

        if normalized < min_qty:
            return 0.0

        if step > 0:
            step_decimals = str(lot.get("qtyStep", "1")).rstrip("0")
            precision = len(step_decimals.split(".")[1]) if "." in step_decimals else 0
            return round(normalized, precision)

        return round(normalized, 6)

    def get_price_tick(self, symbol: str) -> float:
        meta = self._get_symbol_meta(symbol)
        price_filter = meta.get("priceFilter", {}) if meta else {}
        return self._to_float(price_filter.get("tickSize", 0), 0.0)

    def get_price_precision(self, symbol: str) -> int:
        meta = self._get_symbol_meta(symbol)
        price_filter = meta.get("priceFilter", {}) if meta else {}
        tick_text = str(price_filter.get("tickSize", "1")).rstrip("0")
        return len(tick_text.split(".")[1]) if "." in tick_text else 0

    def get_order_rules(self, symbol: str) -> dict:
        meta = self._get_symbol_meta(symbol)
        lot = meta.get("lotSizeFilter", {}) if meta else {}
        price_filter = meta.get("priceFilter", {}) if meta else {}

        tick_size = self._to_float(price_filter.get("tickSize", 0), 0.0)
        qty_step = self._to_float(lot.get("qtyStep", 0), 0.0)
        min_qty = self._to_float(lot.get("minOrderQty", 0), 0.0)
        min_notional = self._to_float(
            lot.get("minNotionalValue", lot.get("minOrderAmt", 0)),
            0.0,
        )

        return {
            "tick_size": tick_size,
            "qty_step": qty_step,
            "min_qty": min_qty,
            "min_notional": min_notional,
            "price_precision": self.get_price_precision(symbol),
        }

    def normalize_price(self, symbol: str, price: float, mode: str = "down") -> float:
        meta = self._get_symbol_meta(symbol)
        price_filter = meta.get("priceFilter", {}) if meta else {}

        tick = self._to_float(price_filter.get("tickSize", 0), 0.0)
        min_price = self._to_float(price_filter.get("minPrice", 0), 0.0)
        max_price = self._to_float(price_filter.get("maxPrice", 0), 0.0)

        normalized = max(0.0, price)

        if tick > 0:
            ratio = normalized / tick
            if mode == "up":
                normalized = math.ceil(ratio) * tick
            elif mode == "nearest":
                normalized = round(ratio) * tick
            else:
                normalized = math.floor(ratio) * tick

        if min_price > 0:
            normalized = max(normalized, min_price)
        if max_price > 0:
            normalized = min(normalized, max_price)

        if tick > 0:
            tick_text = str(price_filter.get("tickSize", "1")).rstrip("0")
            precision = len(tick_text.split(".")[1]) if "." in tick_text else 0
            return round(normalized, precision)

        return round(normalized, 6)
