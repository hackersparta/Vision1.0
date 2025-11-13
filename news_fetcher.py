import os
import csv
import json
import requests
import datetime
import time
import feedparser
import urllib.parse
from trade_db import init_news_table, log_news_sentiment  # ← added import

# ==========================
# CONFIG
# ==========================
BASE_DIR = os.path.dirname(__file__)
STOCK_FILE = os.path.join(BASE_DIR, "my_stocks.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
API_URL = "http://localhost:8000/analyze"

os.makedirs(DATA_DIR, exist_ok=True)
init_news_table()  # ensure table exists before starting

# Load stock list
with open(STOCK_FILE, "r", encoding="utf-8") as f:
    STOCKS = json.load(f)

today = datetime.date.today().isoformat()
csv_file = os.path.join(DATA_DIR, f"sentiment_{today}.csv")

# ==========================
# FUNCTIONS
# ==========================
def fetch_news(stock_name, limit=5):
    """Fetch recent news headlines from Google News RSS."""
    query = urllib.parse.quote_plus(f"{stock_name} stock")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries[:limit]:
        headlines.append({"title": entry.title, "link": entry.link})
    return headlines



def analyze_headline(headline):
    """Send headline to /analyze endpoint."""
    try:
        res = requests.post(API_URL, json={"text": headline}, timeout=15)
        if res.status_code == 200:
            data = res.json()
            return {
                "headline": headline,
                "sentiment": data.get("sentiment"),
                "score": data.get("sentiment_score"),
                "entities": ", ".join(data.get("entities", [])),
                "matched_stocks": ", ".join(data.get("matched_stocks", []))
            }
        else:
            return {"headline": headline, "error": res.text}
    except Exception as e:
        return {"headline": headline, "error": str(e)}

# ==========================
# MAIN
# ==========================
print(f"📰 Starting news fetcher for {len(STOCKS)} stocks...")

with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Date", "Stock", "Headline", "Sentiment", "Confidence", "Entities", "Matched Stocks", "Link"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for stock in STOCKS:
        print(f"\n🔍 Fetching news for {stock}...")
        headlines = fetch_news(stock)
        if not headlines:
            print(f"⚠️ No news found for {stock}")
            continue

        for h in headlines:
            result = analyze_headline(h["title"])
            time.sleep(1)

            # Write to CSV
            writer.writerow({
                "Date": today,
                "Stock": stock,
                "Headline": result.get("headline"),
                "Sentiment": result.get("sentiment"),
                "Confidence": result.get("score"),
                "Entities": result.get("entities"),
                "Matched Stocks": result.get("matched_stocks"),
                "Link": h["link"]
            })

            # Write to DB
            try:
                log_news_sentiment(
                    stock=stock,
                    headline=result.get("headline"),
                    sentiment=result.get("sentiment"),
                    confidence=result.get("score"),
                    entities=result.get("entities"),
                    matched_stocks=result.get("matched_stocks"),
                    link=h["link"]
                )
            except Exception as e:
                print(f"❌ DB insert failed for {stock}: {e}")

        print(f"✅ Completed {stock} ({len(headlines)} headlines)")

print(f"\n📊 Done! Saved to file: {csv_file} and DB: trades.db")
