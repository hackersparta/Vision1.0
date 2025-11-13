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


# ===============================
# NEW SENTIMENT TABLE FUNCTIONS
# ===============================

def init_news_table():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        stock TEXT,
        headline TEXT,
        sentiment TEXT,
        confidence REAL,
        entities TEXT,
        matched_stocks TEXT,
        link TEXT,
        ts TEXT
    )
    """)
    conn.commit()
    conn.close()


def log_news_sentiment(stock, headline, sentiment, confidence, entities, matched_stocks, link):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO news_sentiment
        (date, stock, headline, sentiment, confidence, entities, matched_stocks, link, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().date().isoformat(),
        stock, headline, sentiment, confidence,
        entities, matched_stocks, link, ts
    ))
    conn.commit()
    conn.close()
