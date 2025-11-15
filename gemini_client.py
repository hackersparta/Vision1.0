import os
import requests
import json

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

HEADERS = {
    "x-goog-api-key": GEMINI_API_KEY,
    "Content-Type": "application/json"
}

def analyze_with_gemini(text):
    """
    Sends text to Gemini and returns:
    {
       "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
       "score": 0.00 - 1.00,
       "entities": [],
       "matched_stocks": []
    }
    """
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not found"}

    prompt = f"""
    You are a financial news sentiment engine.
    Analyze the following headline: "{text}"

    Return ONLY valid JSON, no explanation, in this exact structure:
    {{
      "sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
      "score": number (0 to 1),
      "entities": ["list", "of", "ORGs"],
      "matched_stocks": []
    }}

    VERY IMPORTANT: No markdown, no code block, only pure JSON.
    """

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        r = requests.post(API_URL, headers=HEADERS, json=body, timeout=40)
        data = r.json()

        # Extract the text from model response
        raw = data["candidates"][0]["content"]["parts"][0]["text"]

        # Clean if model returned ```json ... ```
        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception as e:
        return {"error": str(e)}
