import time
import logging
import requests
import json

# =====================================================
# SENTIMENT NORMALISATION (Fix for charts)
# =====================================================
def normalize_sentiment(value):
    if not value:
        return None
    v = value.strip().upper()
    if v in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        return v
    return v  # fallback, but uppercase

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    filename="/app/logs/gemini_full.log",
    level=logging.INFO,
    format="%(asctime)s [GEMINI] %(message)s"
)

# =====================================================
# STATIC API KEYS  (REPLACE THESE)
# =====================================================
API_KEYS = [
    "AIzaSyCDHXZFJ6M0gcMXA2ovu4MEkoOpvRZ1xCg",
    "AIzaSyAKPApQ5qPchuQJczPSL5rNb3eC9WWmOaw",
    "AIzaSyCoAuW4KovDa1NmA0aYemhHik8oZ19MJY8",
    "AIzaSyAWyi7N7IuGjH0BIdYDhbDkr4sSobOqmOk",
    "AIzaSyBvTZjBndff6lph3uBYGc95FSML6HOPLFI",
    "AIzaSyCbfbKWKPnJi4unQiN8jD_CraJex4vQimE"
]

# =====================================================
# MODEL FALLBACK ORDER (FREE FLASH MODELS)
# =====================================================
MODEL_LIST = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-flash",
    "gemini-1.0-flash-lite"
]

# Base URL generator
def build_url(model):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# =====================================================
# SEND REQUEST (single API key + single model)
# =====================================================
def _send_request(prompt, api_key, model, source="unknown"):
    url = build_url(model)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=body, timeout=60)
        return res.json()
    except Exception as e:
        return {"error": {"message": f"NETWORK ERROR: {e}"}}


# =====================================================
# TRY MULTIPLE KEYS FOR ONE MODEL
# =====================================================
def _try_keys_for_model(prompt, model, source):
    for key_index, key in enumerate(API_KEYS):
        logging.info(f"[{source}] Using API KEY {key_index+1}/{len(API_KEYS)} for model {model}")

        result = _send_request(prompt, key, model, source)

        # No error → valid success
        if "error" not in result:
            return result

        # Error case
        msg = result["error"].get("message", "UNKNOWN ERROR")
        logging.error(f"[{source}] Model {model} — Key {key_index+1} FAILED → {msg}")

        # Try next key
        continue

    return {"error": {"message": f"All keys failed for model {model}"}}


# =====================================================
# MASTER FALLBACK: TRY ALL MODELS × ALL KEYS
# =====================================================
def _try_models_and_keys(prompt, source):
    for model_index, model in enumerate(MODEL_LIST):
        logging.info(f"[{source}] Trying MODEL {model_index+1}/{len(MODEL_LIST)} → {model}")

        result = _try_keys_for_model(prompt, model, source)

        if "error" not in result:
            logging.info(f"[{source}] SUCCESS with model {model}")
            return result

        error_msg = result["error"]["message"]
        logging.error(f"[{source}] Model FAILED ({model}) → {error_msg}")

        # Continue to next model
        continue

    logging.critical(f"[{source}] ALL MODELS + ALL KEYS FAILED!")
    return {"error": {"message": "ALL MODELS + ALL KEYS FAILED"}}


# =====================================================
# PUBLIC: SINGLE HEADLINE
# =====================================================
def analyze_with_gemini(text, source="single", retries=2):

    prompt = f"""
You are a financial news sentiment engine.
Analyze the following headline:

\"{text}\"

Return ONLY JSON:
{{
  "sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
  "score": 0.xx,
  "entities": [],
  "matched_stocks": []
}}
"""

    for attempt in range(1, retries+1):
        logging.info(f"[{source}] ATTEMPT {attempt}")

        data = _try_models_and_keys(prompt, source)

        if "error" in data:
            logging.error(f"[{source}] FAILED: {data['error']['message']}")
            time.sleep(1)
            continue

        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)

            if "sentiment" in parsed:
                parsed["sentiment"] = normalize_sentiment(parsed["sentiment"])

            logging.info(f"[{source}] FINAL JSON:\n{json.dumps(parsed, indent=2)}")
            return parsed

        except Exception as e:
            logging.error(f"[{source}] PARSE ERROR: {e}")
            # log raw content for debugging
            try:
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                logging.error(f"[{source}] RAW RESPONSE:\n{raw}")
            except Exception:
                pass
            continue

    logging.error(f"[{source}] FAILED AFTER RETRIES")
    return {"error": "Failed after retries"}


# =====================================================
# PUBLIC: BATCH HEADLINES
# =====================================================
def batch_analyze_with_gemini(headlines, source="batch", retries=2):

    numbered = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])

    prompt = f"""
You are a financial news sentiment engine.
Analyze ALL the following headlines:

{numbered}

Return JSON ARRAY ONLY:
[
  {{
    "sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
    "score": 0.xx,
    "entities": [],
    "matched_stocks": []
  }},
  ...
]

NO markdown, no explanation.
"""

    for attempt in range(1, retries+1):
        logging.info(f"[{source}] BATCH ATTEMPT {attempt} ({len(headlines)} headlines)")

        data = _try_models_and_keys(prompt, source)

        if "error" in data:
            logging.error(f"[{source}] BATCH FAILED: {data['error']['message']}")
            time.sleep(1)
            continue

        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)

            # normalize sentiment labels
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "sentiment" in item:
                        item["sentiment"] = normalize_sentiment(item["sentiment"])
            elif isinstance(parsed, dict):
                if "sentiment" in parsed:
                    parsed["sentiment"] = normalize_sentiment(parsed["sentiment"])
                parsed = [parsed]

            # Ensure same length
            if len(parsed) != len(headlines):
                logging.warning(f"[{source}] LENGTH MISMATCH: parsed={len(parsed)} expected={len(headlines)}")

            for i, item in enumerate(parsed, start=1):
                logging.info(f"[{source}] ITEM {i}/{len(parsed)}:\n{json.dumps(item, indent=2)}")

            return parsed

        except Exception as e:
            logging.error(f"[{source}] PARSE ERROR: {e}")
            try:
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                logging.error(f"[{source}] RAW RESPONSE:\n{raw}")
            except Exception:
                pass
            continue

    logging.error(f"[{source}] BATCH FAILED AFTER RETRIES")
    return [None] * len(headlines)


# =====================================================
# NEW: CLASSIFY IF HEADLINE IS FINANCIAL (YES/NO)
# =====================================================
def classify_financial(text, source="is_financial", retries=2):
    """
    Returns True if the headline is financial / likely to affect stock price, else False.
    The function asks the model to return only JSON: {"is_financial":"YES" | "NO"}
    """

    prompt = f"""
You are a classifier that decides whether a short news headline is relevant to stock prices or company/market financial performance.
Return ONLY JSON with a single field.

Example responses:
{{"is_financial":"YES"}}
{{"is_financial":"NO"}}

Analyze this headline:
\"{text}\"

Question: Does this headline have a plausible impact on stock price or company financials? (Answer YES or NO)
"""

    for attempt in range(1, retries+1):
        logging.info(f"[{source}] ATTEMPT {attempt} for text: {text[:120]}")

        data = _try_models_and_keys(prompt, source)

        if "error" in data:
            logging.error(f"[{source}] FAILED: {data['error']['message']}")
            time.sleep(1)
            continue

        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(cleaned)
            val = parsed.get("is_financial", "").strip().upper()

            logging.info(f"[{source}] CLASSIFY -> {val}")
            return val == "YES"

        except Exception as e:
            logging.error(f"[{source}] PARSE ERROR: {e}")
            try:
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                logging.error(f"[{source}] RAW RESPONSE:\n{raw}")
            except Exception:
                pass
            continue

    logging.error(f"[{source}] FAILED AFTER RETRIES")
    return False
