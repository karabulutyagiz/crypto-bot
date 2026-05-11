import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Optional, cast
from config import TIMEZONE_OFFSET

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit1 REAL NOT NULL,
            take_profit2 REAL,
            take_profit3 REAL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            tp1_hit INTEGER DEFAULT 0,
            partial_closed_size REAL DEFAULT 0,
            strategy TEXT,
            notes TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(trades)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "take_profit3" not in existing_cols:
        cursor.execute("ALTER TABLE trades ADD COLUMN take_profit3 REAL")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            total_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            extra TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_trade(
    symbol: str,
    side: str,
    entry_price: float,
    size: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: float,
    strategy: str = "ema_rsi_macd"
):
    conn = get_connection()
    cursor = conn.cursor()
    
    now = (datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)).isoformat()
    
    cursor.execute("""
        INSERT INTO trades (
            symbol, side, entry_price, size, stop_loss,
            take_profit1, take_profit2, take_profit3,
            entry_time, status, strategy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
    """, (symbol, side, entry_price, size, stop_loss, tp1, tp2, tp3, now, strategy))
    
    trade_id = int(cursor.lastrowid or 0)
    conn.commit()
    conn.close()
    
    return trade_id

def update_trade(
    trade_id: int,
    exit_price=None,
    pnl=None,
    status=None,
    tp1_hit=None,
    partial_closed_size=None
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if exit_price is not None:
        updates.append("exit_price = ?")
        params.append(exit_price)
    if pnl is not None:
        updates.append("pnl = ?")
        params.append(pnl)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
        if status == "closed":
            now = (datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)).isoformat()
            updates.append("exit_time = ?")
            params.append(now)
    if tp1_hit is not None:
        updates.append("tp1_hit = ?")
        params.append(1 if tp1_hit else 0)
    if partial_closed_size is not None:
        updates.append("partial_closed_size = ?")
        params.append(partial_closed_size)
    
    if updates:
        params.append(trade_id)
        cursor.execute(f"""
            UPDATE trades SET {', '.join(updates)} WHERE id = ?
        """, params)
        conn.commit()
    
    conn.close()

def get_open_trade(symbol: str, side: Optional[str] = None) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    if side:
        cursor.execute(
            """
            SELECT *
            FROM trades
            WHERE symbol = ? AND side = ? AND status = 'open'
            ORDER BY datetime(entry_time) DESC, id DESC
            LIMIT 1
            """,
            (symbol, side),
        )
    else:
        cursor.execute(
            """
            SELECT *
            FROM trades
            WHERE symbol = ? AND status = 'open'
            ORDER BY datetime(entry_time) DESC, id DESC
            LIMIT 1
            """,
            (symbol,),
        )
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None

def get_trade_history(limit: int = 100) -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def update_daily_stats(date: str, pnl: float, is_win: bool) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO daily_stats (date, total_trades, wins, losses, total_pnl)
        VALUES (?, 1, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            total_trades = total_trades + 1,
            wins = wins + ?,
            losses = losses + ?,
            total_pnl = total_pnl + ?
    """, (date, 1 if is_win else 0, 0 if is_win else 1, pnl,
          1 if is_win else 0, 0 if is_win else 1, pnl))
    
    conn.commit()
    conn.close()

def get_daily_stats(date=None) -> dict:
    if date is None:
        date = (datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)).strftime("%Y-%m-%d")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM daily_stats WHERE date = ?
    """, (date,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return {"date": date, "total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0}

def get_all_time_stats() -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl
        FROM trades WHERE status = 'closed'
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else {}

init_db()
