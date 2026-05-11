from typing import Dict, List, Tuple, Optional
from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET, BYBIT_DEMO_TRADING
from logger import logger


class OrderBookAnalyzer:
    def __init__(self):
        self.session = HTTP(
            testnet=BYBIT_TESTNET,
            demo=BYBIT_DEMO_TRADING,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
        )
    
    def get_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict]:
        try:
            response = self.session.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            if response["retCode"] == 0:
                return response["result"]
            return None
        except Exception as e:
            logger.error(f"Order book hatası: {e}")
            return None
    
    def analyze_order_book(self, symbol: str) -> Dict:
        ob = self.get_order_book(symbol)
        if not ob:
            return {"error": "Order book alınamadı"}
        
        bids = ob.get("b", [])
        asks = ob.get("a", [])
        
        if not bids or not asks:
            return {"error": "Yetersiz veri"}
        
        bids = [(float(b[0]), float(b[1])) for b in bids]
        asks = [(float(a[0]), float(a[1])) for a in asks]
        
        total_bid_volume = sum(b[1] for b in bids)
        total_ask_volume = sum(a[1] for a in asks)
        
        bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1
        
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        spread_percent = (spread / best_bid * 100) if best_bid > 0 else 0
        
        bid_walls = self._find_walls(bids, total_bid_volume)
        ask_walls = self._find_walls(asks, total_ask_volume)
        
        bid_imbalance, ask_imbalance = self._calculate_imbalance(bids, asks, 5)
        
        return {
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_percent": spread_percent,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "bid_ask_ratio": bid_ask_ratio,
            "bid_walls": bid_walls,
            "ask_walls": ask_walls,
            "bid_imbalance": bid_imbalance,
            "ask_imbalance": ask_imbalance,
            "sentiment": self._get_sentiment(bid_ask_ratio, bid_imbalance, ask_imbalance)
        }
    
    def _find_walls(self, orders: List[Tuple], total_volume: float, threshold: float = 0.2) -> List[Dict]:
        walls = []
        for price, volume in orders:
            if volume / total_volume > threshold:
                walls.append({"price": price, "volume": volume, "percent": volume / total_volume * 100})
        return walls
    
    def _calculate_imbalance(self, bids: List[Tuple], asks: List[Tuple], depth: int = 5) -> Tuple[float, float]:
        top_bids = bids[:depth]
        top_asks = asks[:depth]
        
        bid_value = sum(p * v for p, v in top_bids)
        ask_value = sum(p * v for p, v in top_asks)
        
        total = bid_value + ask_value
        if total == 0:
            return 0.5, 0.5
        
        bid_imbalance = bid_value / total
        ask_imbalance = ask_value / total
        
        return bid_imbalance, ask_imbalance
    
    def _get_sentiment(self, ratio: float, bid_imb: float, ask_imb: float) -> str:
        if ratio > 1.5 and bid_imb > 0.6:
            return "bullish"
        elif ratio < 0.67 and ask_imb > 0.6:
            return "bearish"
        return "neutral"
    
    def check_support_resistance(self, symbol: str, current_price: float) -> Dict:
        ob = self.get_order_book(symbol, limit=200)
        if not ob:
            return {"error": "Order book alınamadı"}
        
        bids = [(float(b[0]), float(b[1])) for b in ob.get("b", [])]
        asks = [(float(a[0]), float(a[1])) for a in ob.get("a", [])]
        
        support_levels = []
        resistance_levels = []
        
        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)
        
        for price, volume in bids:
            if volume / total_bid_vol > 0.05 and price < current_price:
                support_levels.append({"price": price, "strength": volume / total_bid_vol * 100})
        
        for price, volume in asks:
            if volume / total_ask_vol > 0.05 and price > current_price:
                resistance_levels.append({"price": price, "strength": volume / total_ask_vol * 100})
        
        support_levels = sorted(support_levels, key=lambda x: x["strength"], reverse=True)[:3]
        resistance_levels = sorted(resistance_levels, key=lambda x: x["strength"], reverse=True)[:3]
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "supports": support_levels,
            "resistances": resistance_levels
        }
    
    def get_market_pressure(self, symbol: str) -> Dict:
        analysis = self.analyze_order_book(symbol)
        if "error" in analysis:
            return analysis
        
        sentiment = analysis["sentiment"]
        ratio = analysis["bid_ask_ratio"]
        
        pressure_score = 0
        if sentiment == "bullish":
            pressure_score = min(100, (ratio - 1) * 50 + 50)
        elif sentiment == "bearish":
            pressure_score = max(-100, (1 - ratio) * -50 - 50)
        else:
            pressure_score = (ratio - 1) * 50
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "pressure_score": pressure_score,
            "bid_ask_ratio": ratio,
            "interpretation": self._interpret_pressure(pressure_score)
        }
    
    def _interpret_pressure(self, score: float) -> str:
        if score >= 75:
            return "Çok güçlü alış baskısı"
        elif score >= 50:
            return "Güçlü alış baskısı"
        elif score >= 25:
            return "Hafif alış baskısı"
        elif score >= -25:
            return "Nötr"
        elif score >= -50:
            return "Hafif satış baskısı"
        elif score >= -75:
            return "Güçlü satış baskısı"
        else:
            return "Çok güçlü satış baskısı"


orderbook_analyzer = OrderBookAnalyzer()
