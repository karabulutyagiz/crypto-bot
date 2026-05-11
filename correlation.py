from typing import Dict, List, Set

SECTOR_MAPPING: Dict[str, str] = {
    "BTCUSDT": "majors",
    "ETHUSDT": "majors",
    "BNBUSDT": "majors",
    "SOLUSDT": "l1",
    "ADAUSDT": "l1",
    "AVAXUSDT": "l1",
    "MATICUSDT": "l1",
    "DOTUSDT": "l1",
    "ATOMUSDT": "cosmos",
    "INJUSDT": "cosmos",
    "TIAUSDT": "cosmos",
    "OSMOUSDT": "cosmos",
    "LINKUSDT": "oracle",
    "TRBUSDT": "oracle",
    "DOGEUSDT": "meme",
    "SHIBUSDT": "meme",
    "PEPEUSDT": "meme",
    "BONKUSDT": "meme",
    "WIFUSDT": "meme",
    "FLOKIUSDT": "meme",
    "XRPUSDT": "payment",
    "XLMUSDT": "payment",
    "LTCUSDT": "legacy",
    "BCHUSDT": "legacy",
    "ETCUSDT": "legacy",
    "AAVEUSDT": "defi",
    "COMPUSDT": "defi",
    "UNIUSDT": "defi",
    "SUSHIUSDT": "defi",
    "CRVUSDT": "defi",
    "MKRUSDT": "defi",
    "SNXUSDT": "defi",
    "GMXUSDT": "defi",
    "DYDXUSDT": "defi",
    "ARBUSDT": "l2",
    "OPUSDT": "l2",
    "MNTUSDT": "l2",
    "STRKUSDT": "l2",
    "NEARUSDT": "l1",
    "FTMUSDT": "l1",
    "SEIUSDT": "l1",
    "SUIUSDT": "l1",
    "APTUSDT": "l1",
    "BLURUSDT": "nft",
    "APEUSDT": "nft",
    "IMXUSDT": "nft",
    "MASKUSDT": "social",
    "GALUSDT": "identity",
    "LDOUSDT": "eth_ecosystem",
    "ENSUSDT": "eth_ecosystem",
    "RNDRUSDT": "ai",
    "FETUSDT": "ai",
    "AGIXUSDT": "ai",
    "TAOUSDT": "ai",
    "GRTUSDT": "indexing",
}

EXCLUDED_SECTORS: Set[str] = set()

class CorrelationManager:
    def __init__(self):
        self.active_sectors: Set[str] = set()
        self.active_symbols: Set[str] = set()
    
    def get_sector(self, symbol: str) -> str:
        base_symbol = symbol.replace("USDT", "").replace("USDC", "") + "USDT"
        return SECTOR_MAPPING.get(base_symbol, SECTOR_MAPPING.get(symbol, "unknown"))
    
    def can_trade(self, symbol: str) -> tuple:
        if symbol in self.active_symbols:
            return False, f"{symbol} zaten açık pozisyonda"
        
        sector = self.get_sector(symbol)
        
        if sector in EXCLUDED_SECTORS:
            return False, f"{sector} sektörü excluded listesinde"
        
        if sector in self.active_sectors and sector != "unknown":
            return False, f"{sector} sektöründe zaten açık pozisyon var"
        
        return True, "İşlem açılabilir"
    
    def add_position(self, symbol: str) -> None:
        sector = self.get_sector(symbol)
        self.active_sectors.add(sector)
        self.active_symbols.add(symbol)
    
    def remove_position(self, symbol: str) -> None:
        self.active_symbols.discard(symbol)
        self.active_sectors = {self.get_sector(sym) for sym in self.active_symbols}
    
    def get_sector_symbols(self, sector: str) -> List[str]:
        return [s for s, sct in SECTOR_MAPPING.items() if sct == sector]
    
    def get_correlated_symbols(self, symbol: str) -> List[str]:
        sector = self.get_sector(symbol)
        if sector == "unknown":
            return []
        return [s for s in self.get_sector_symbols(sector) if s != symbol]

correlation_manager = CorrelationManager()
