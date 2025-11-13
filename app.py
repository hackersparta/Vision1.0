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
        return render_template("sentiment.html", data=rows)
    except Exception as e:
        return f"Error loading sentiment data: {e}"

@app.route("/sentiment_data")
def sentiment_data():
    """Returns summarized sentiment data for charts"""
    import sqlite3
    from trade_db import DB

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Count of sentiments (for Pie Chart)
    cur.execute("""
        SELECT sentiment, COUNT(*) FROM news_sentiment
        GROUP BY sentiment
    """)
    sentiment_counts = dict(cur.fetchall())

    # Top 5 stocks with most positive sentiment (for Bar Chart)
    cur.execute("""
        SELECT stock, COUNT(*) as total
        FROM news_sentiment
        WHERE sentiment = 'POSITIVE'
        GROUP BY stock
        ORDER BY total DESC
        LIMIT 5
    """)
    top_positive = [{"stock": row[0], "count": row[1]} for row in cur.fetchall()]

    conn.close()
    return jsonify({
        "sentiment_counts": sentiment_counts,
        "top_positive": top_positive
    })

    @app.route("/run_news_fetcher", methods=["POST"])
    def run_news_fetcher():
    """Manually trigger the news_fetcher script"""
    try:
        subprocess.Popen(["python3", "news_fetcher.py"])
        return jsonify({"status": "✅ News fetcher started! Check logs in terminal."})
    except Exception as e:
        return jsonify({"status": f"❌ Failed to start news fetcher: {e}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
    