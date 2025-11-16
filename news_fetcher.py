import os
import csv
import json
import datetime
import time
import feedparser
import urllib.parse
import difflib
import logging
from trade_db import init_news_table, log_news_sentiment
import argparse
from gemini_client import batch_analyze_with_gemini, classify_financial

# ----------------------------
# CLI args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--stocks")
parser.add_argument("--days")
args = parser.parse_args()

# ==========================
# CONFIG
# ==========================
BASE_DIR = os.path.dirname(__file__)
STOCK_FILE = os.path.join(BASE_DIR, "my_stocks.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
init_news_table()

# Load stock list
with open(STOCK_FILE, "r", encoding="utf-8") as f:
    all_stocks = json.load(f)

if args.stocks:
    STOCKS = args.stocks.split(",")
else:
    STOCKS = all_stocks

fetch_days = int(args.days) if args.days else 1
today = datetime.date.today().isoformat()
csv_file = os.path.join(DATA_DIR, f"sentiment_{today}.csv")

# Logging: mixed readable + JSON structured
LOG_PATH = os.path.join(LOG_DIR, f"news_fetcher_{today}.log")
logger = logging.getLogger("news_fetcher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [NEWS_FETCHER] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# ==========================
# RSS TEMPLATES (SEARCH)
# - Google News search RSS (regional)
# - Yahoo search RSS
# - Site-specific Google filters for MoneyControl, ET, Reuters
# ==========================
RSS_TEMPLATES = [
    # Generic Google News search (India)
    "https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en",
    # Yahoo search RSS
    "https://news.search.yahoo.com/rss?p={query}",
    # MoneyControl via Google site filter
    "https://news.google.com/rss/search?q={query}+site:moneycontrol.com+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en",
    # Economic Times via Google site filter
    "https://news.google.com/rss/search?q={query}+site:economictimes.indiatimes.com+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en",
    # Reuters via Google site filter (global)
    "https://news.google.com/rss/search?q={query}+site:reuters.com+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en"
]


# ==========================
# HELPERS
# ==========================
def fetch_rss(url):
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            # still may contain entries; log bozo for debugging
            logger.info(json.dumps({"type": "rss_bozo", "url": url, "bozo": True}))
        return feed.entries or []
    except Exception as e:
        logger.error(json.dumps({"type": "rss_error", "url": url, "error": str(e)}))
        return []


def fetch_news_for_source(stock_name, template, limit=15, days=1):
    query = urllib.parse.quote_plus(f"{stock_name} stock")
    url = template.format(query=query, days=days)
    entries = fetch_rss(url)
    headlines = []
    for entry in entries[:limit]:
        title = getattr(entry, "title", None) or getattr(entry, "summary", None) or ""
        link = getattr(entry, "link", None) or ""
        if title:
            headlines.append({"title": title.strip(), "link": link})
    return headlines


def fetch_news(stock_name, per_source_limit=6):
    """
    Fetch headlines from multiple RSS templates, merge and dedupe.
    Returns list of dicts: {"title":..., "link":...}
    """
    all_headlines = []
    for tmpl in RSS_TEMPLATES:
        try:
            h = fetch_news_for_source(stock_name, tmpl, limit=per_source_limit, days=fetch_days)
            if h:
                all_headlines.extend(h)
                # structured log for fetched source
                logger.info(json.dumps({
                    "type": "fetched_source",
                    "stock": stock_name,
                    "template": tmpl,
                    "count": len(h)
                }))
        except Exception as e:
            logger.error(json.dumps({"type": "fetch_error", "stock": stock_name, "template": tmpl, "error": str(e)}))

    # dedupe with similarity threshold
    unique = []
    for item in all_headlines:
        title = item["title"]
        dup = False
        for u in unique:
            ratio = difflib.SequenceMatcher(None, title.lower(), u["title"].lower()).ratio()
            if ratio >= 0.85:
                dup = True
                # prefer having link if it exists
                if not u.get("link") and item.get("link"):
                    u["link"] = item.get("link")
                break
        if not dup:
            unique.append(item)

    logger.info(json.dumps({
        "type": "dedupe_summary",
        "stock": stock_name,
        "raw_count": len(all_headlines),
        "unique_count": len(unique),
        "dedupe_threshold": 0.85
    }))

    return unique


# ==========================
# MAIN
# ==========================
print("News fetcher started")
logger.info(json.dumps({"type": "start", "date": today, "stock_count": len(STOCKS)}))

with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Date", "Stock", "Headline", "Sentiment",
        "Confidence", "Entities", "Matched Stocks", "Link"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for stock in STOCKS:
        try:
            # Minimal console output
            print(f"Fetching news for {stock}...")

            raw_headlines = fetch_news(stock, per_source_limit=8)
            if not raw_headlines:
                logger.info(json.dumps({"type": "no_news", "stock": stock}))
                print(f"Finished {stock}")
                continue

            # Financial filter (Gemini)
            filtered = []
            classified = []
            for h in raw_headlines:
                title = h["title"]
                try:
                    is_fin = classify_financial(title, source="fetch_filter")
                except Exception as e:
                    logger.error(json.dumps({"type": "classify_error", "stock": stock, "headline": title[:140], "error": str(e)}))
                    is_fin = True  # fail-open

                classified.append({"title": title, "link": h.get("link"), "is_financial": is_fin})
                if is_fin:
                    filtered.append(h)

            logger.info(json.dumps({
                "type": "classification_summary",
                "stock": stock,
                "raw_count": len(raw_headlines),
                "filtered_count": len(filtered)
            }))

            if not filtered:
                logger.info(json.dumps({"type": "no_filtered_news", "stock": stock}))
                print(f"Finished {stock}")
                continue

            titles = [h["title"] for h in filtered]

            # Batch analysis
            batch_size = 15
            batch_no = 0
            for i in range(0, len(titles), batch_size):
                batch_no += 1
                batch = titles[i:i + batch_size]
                # silence to file, minimal to console
                logger.info(json.dumps({
                    "type": "batch_start",
                    "stock": stock,
                    "batch_no": batch_no,
                    "batch_size": len(batch)
                }))

                results = batch_analyze_with_gemini(batch, source="fetch_batch")

                # Write CSV + DB for each parsed item
                processed_count = 0
                for idx, parsed in enumerate(results):
                    title = batch[idx] if idx < len(batch) else "(missing)"
                    if not parsed or "sentiment" not in parsed:
                        logger.info(json.dumps({"type": "skip_parsed", "stock": stock, "title": title[:140]}))
                        continue

                    sentiment = parsed.get("sentiment")
                    score = parsed.get("score") or 0
                    entities = ", ".join(parsed.get("entities", [])) if parsed.get("entities") else ""
                    matched = ", ".join(parsed.get("matched_stocks", [])) if parsed.get("matched_stocks") else ""
                    link = filtered[i + idx]["link"] if (i + idx) < len(filtered) else ""

                    writer.writerow({
                        "Date": today,
                        "Stock": stock,
                        "Headline": title,
                        "Sentiment": sentiment,
                        "Confidence": score,
                        "Entities": entities,
                        "Matched Stocks": matched,
                        "Link": link
                    })

                    try:
                        log_news_sentiment(
                            stock=stock,
                            headline=title,
                            sentiment=sentiment,
                            confidence=score,
                            entities=entities,
                            matched_stocks=matched,
                            link=link
                        )
                    except Exception as e:
                        logger.error(json.dumps({"type": "db_error", "stock": stock, "error": str(e)}))

                    # structured log for each result (JSON)
                    logger.info(json.dumps({
                        "type": "news_item",
                        "stock": stock,
                        "headline": title,
                        "sentiment": sentiment,
                        "score": score,
                        "entities": parsed.get("entities", []),
                        "matched_stocks": parsed.get("matched_stocks", []),
                        "link": link
                    }))

                    processed_count += 1

                # Console minimal: batch processed
                print(f"Batch {batch_no} processed")
                logger.info(json.dumps({
                    "type": "batch_end",
                    "stock": stock,
                    "batch_no": batch_no,
                    "processed_count": processed_count
                }))

                # polite pause
                time.sleep(0.3)

            print(f"Finished {stock}")
            logger.info(json.dumps({"type": "stock_done", "stock": stock, "batches": batch_no}))

        except Exception as e:
            # Print minimal to console; full details to log
            print(f"ERROR fetching {stock}: {str(e)}")
            logger.error(json.dumps({"type": "unexpected_error", "stock": stock, "error": str(e)}))

logger.info(json.dumps({"type": "complete", "date": today, "csv": csv_file}))
print("News fetcher completed")
