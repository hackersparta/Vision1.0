# app.py
from flask import Flask, request, jsonify, render_template
import random
from datetime import datetime
from trade_db import init_db, log_trade, list_trades

app = Flask(__name__)

# Initialize database
init_db()


# ----------------------------
# 1. Simple momentum algorithm
# ----------------------------
def simple_momentum_signal(prices, short_window=3, long_window=7):
    """Basic moving-average crossover rule"""
    if len(prices) < long_window:
        return "HOLD", "not enough data"

    short_avg = sum(prices[-short_window:]) / short_window
    long_avg = sum(prices[-long_window:]) / long_window

    if short_avg > long_avg * 1.001:
        return "BUY", f"short_avg {short_avg:.4f} > long_avg {long_avg:.4f}"
    elif short_avg < long_avg * 0.999:
        return "SELL", f"short_avg {short_avg:.4f} < long_avg {long_avg:.4f}"
    else:
        return "HOLD", "averages near equal"


# ----------------------------
# 2. Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/signal", methods=["POST"])
def signal():
    """
    POST JSON:
      { "prices": [123.4, 124.0, ...] }
    """
    try:
        body = request.json or {}
        prices = body.get("prices")

        if not prices or not isinstance(prices, list):
            base = 100 + random.random() * 10
            prices = [base + (i * 0.1) for i in range(10)]

        side, reason = simple_momentum_signal(prices)
        return jsonify({"signal": side, "reason": reason, "last_price": prices[-1]})

    except Exception as e:
        return jsonify({"signal": "HOLD", "reason": f"error: {e}"})


@app.route("/paper_trade", methods=["POST"])
def paper_trade():
    """
    POST JSON:
      { "side": "BUY"|"SELL", "quantity": 1.0, "price": 123.4, "reason": "manual" }
    """
    data = request.json or {}
    side = data.get("side")
    qty = float(data.get("quantity", 1.0))
    price = data.get("price")
    reason = data.get("reason", "manual")

    if side not in ("BUY", "SELL"):
        return jsonify({"status": "error", "message": "Invalid side"}), 400

    log_trade(side, qty, price, reason)
    return jsonify({"status": "ok", "message": f"{side} logged"})


@app.route("/trades", methods=["GET"])
def trades_api():
    rows = list_trades(100)
    keys = ["id", "side", "quantity", "price", "reason", "ts"]
    trades = [dict(zip(keys, r)) for r in rows]
    return jsonify({"trades": trades})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
