# trade_db.py
import sqlite3
from datetime import datetime

DB = "trades.db"


def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        side TEXT,
        quantity REAL,
        price REAL,
        reason TEXT,
        ts TEXT
    )
    """)
    conn.commit()
    conn.close()


def log_trade(side, quantity, price=None, reason="paper"):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO trades (side, quantity, price, reason, ts) VALUES (?, ?, ?, ?, ?)",
        (side, quantity, price, reason, ts),
    )
    conn.commit()
    conn.close()


def list_trades(limit=100):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, side, quantity, price, reason, ts FROM trades ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
