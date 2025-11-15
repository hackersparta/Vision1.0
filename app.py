import os
import json
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
from trade_db import DB
import sqlite3
import subprocess

app = Flask(__name__)
CORS(app)

# ================== LOAD NLP MODELS ==================
print("Loading NLP models...")

# Lightweight sentiment & NER pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
ner_model = pipeline("ner", grouped_entities=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# ================== STOCK LIST ==================
# ================== STOCK LIST ==================
STOCK_LIST_PATH = os.path.join(os.path.dirname(__file__), "my_stocks.json")

# Load your existing stock list
with open(STOCK_LIST_PATH, "r") as f:
    STOCK_LIST = [s.lower() for s in json.load(f)]

# ================== ROUTES ==================

@app.route('/')
def home():
    return render_template('index.html') if os.path.exists("templates/index.html") else jsonify({
        "message": "App running! Use /analyze endpoint for NLP analysis."
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        user_input = request.json.get('text', '').strip()
        if not user_input:
            return jsonify({'error': 'No text provided.'}), 400

        # Sentiment analysis
        sentiment_result = sentiment_analyzer(user_input)[0]
        sentiment = sentiment_result['label']
        sentiment_score = round(sentiment_result['score'], 3)

        # Named Entity Recognition
        entities = ner_model(user_input)
        extracted_entities = [e['word'] for e in entities if e['entity_group'] == 'ORG']

        # Match entities with your stock list
        matched_stocks = [
            stock for stock in STOCK_LIST
            if any(stock.lower() in entity.lower() for entity in extracted_entities)
        ]

        # Final result
        result = {
            "text": user_input,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "entities": extracted_entities,
            "matched_stocks": matched_stocks
        }

        return jsonify(result)

    except Exception as e:
        print("Error in /analyze:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/sentiment")
def sentiment_page():
    """Render HTML dashboard showing latest sentiment analysis"""
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT date, stock, headline, sentiment, confidence, entities, link
            FROM news_sentiment
            ORDER BY id DESC LIMIT 100
        """)
        rows = cur.fetchall()
        conn.close()

        # Load full stock list from JSON
        with open("my_stocks.json", "r") as f:
            full_stock_list = json.load(f)

        return render_template(
            "sentiment.html",
            data=rows,
            stocks=full_stock_list
        )

    except Exception as e:
        return f"Error loading sentiment data: {e}"

@app.route("/sentiment_data")
@app.route("/sentiment_data")
@app.route("/sentiment_data")
def sentiment_data():
    import sqlite3
    from trade_db import DB
    from flask import request
    from datetime import datetime, timedelta

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    stock = request.args.get("stock")
    days = request.args.get("days", type=int, default=7)
    mode = request.args.get("mode", "all")

    # Get latest entry date
    cur.execute("SELECT date FROM news_sentiment ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    latest_date = row[0] if row else None

    where_clause = []
    params = []

    # ----- CURRENT FETCH FILTER -----
    if mode == "current" and latest_date:
        where_clause.append("date LIKE ?")
        params.append(latest_date.split("T")[0] + "%")

    # ----- HISTORICAL FILTER -----
    else:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        where_clause.append("substr(date, 1, 10) >= ?")
        params.append(start_date)

    if stock and stock.lower() != "all":
        where_clause.append("LOWER(stock)=LOWER(?)")
        params.append(stock)

    where_sql = "WHERE " + " AND ".join(where_clause)

    # --- Sentiment counts ---
    cur.execute(f"""
        SELECT sentiment, COUNT(*) FROM news_sentiment
        {where_sql}
        GROUP BY sentiment
    """, params)
    sentiment_counts_raw = dict(cur.fetchall())
    sentiment_counts = {k: v for k, v in sentiment_counts_raw.items() if k}


    # --- Top positive ---
    cur.execute(f"""
        SELECT stock, COUNT(*) FROM news_sentiment
        {where_sql} AND sentiment='POSITIVE'
        GROUP BY stock
        ORDER BY COUNT(*) DESC LIMIT 5
    """, params)
    top_positive = [{"stock": s, "count": c} for s,c in cur.fetchall()]

    # --- Table rows ---
    cur.execute(f"""
        SELECT date, stock, headline, sentiment, confidence, entities, link
        FROM news_sentiment
        {where_sql}
        ORDER BY id DESC LIMIT 100
    """, params)
    rows = cur.fetchall()

    conn.close()

    return jsonify({
        "sentiment_counts": sentiment_counts,
        "top_positive": top_positive,
        "rows": rows,
        "latest_date": latest_date
    })

@app.route("/run_news_fetcher", methods=["POST"])
def run_news_fetcher():
    """Run news fetcher with selected stocks + days"""
    import subprocess
    import time
    import json

    data = request.get_json()  # get JSON from frontend
    selected_stocks = data.get("stocks", [])
    days = data.get("days", 1)

    # base command
    cmd = ["python3", "news_fetcher.py"]

    # add selected stocks
    if selected_stocks and len(selected_stocks) > 0:
        cmd += ["--stocks", ",".join(selected_stocks)]

    # add days argument
    if days:
        cmd += ["--days", str(days)]

    try:
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True)
        duration = round(time.time() - start_time, 2)

        if process.returncode == 0:
            return jsonify({
                "status": f"✅ Completed in {duration} sec",
                "output": process.stdout[-2000:]
            })
        else:
            return jsonify({
                "status": f"❌ Failed in {duration} sec",
                "error": process.stderr
            })

    except Exception as e:
        return jsonify({"status": f"❌ Error: {e}"})


@app.route("/sentiment_trend")
def sentiment_trend():
    import sqlite3
    from trade_db import DB
    from flask import request
    from datetime import datetime, timedelta

    stock = request.args.get("stock", "all")
    days = int(request.args.get("days", 7))

    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    if stock.lower() == "all":
        cur.execute("""
            SELECT substr(date,1,10) as d,
                   AVG(CASE 
                        WHEN sentiment='POSITIVE' THEN confidence
                        WHEN sentiment='NEGATIVE' THEN -confidence
                        ELSE 0
                       END)
            FROM news_sentiment
            WHERE substr(date,1,10) >= ?
            GROUP BY d ORDER BY d ASC
        """, (start_date,))
    else:
        cur.execute("""
            SELECT substr(date,1,10) as d,
                   AVG(CASE 
                        WHEN sentiment='POSITIVE' THEN confidence
                        WHEN sentiment='NEGATIVE' THEN -confidence
                        ELSE 0
                       END)
            FROM news_sentiment
            WHERE substr(date,1,10) >= ?
              AND LOWER(stock)=LOWER(?)
            GROUP BY d ORDER BY d ASC
        """, (start_date, stock))

    rows = cur.fetchall()
    conn.close()

    return jsonify({
        "dates": [r[0] for r in rows],
        "scores": [float(r[1]) for r in rows]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    
