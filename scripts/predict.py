# predict.py

from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from pathlib import Path

app = Flask(__name__)

# -------------------------------------------------------------------
# Paths and model loading
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cardio_model.joblib"
FEATURES_PATH = BASE_DIR / "cardio_feature_columns.joblib"
INDEX_HTML_PATH = BASE_DIR / "index.html"

# Load model + feature list
model = load(MODEL_PATH)
feature_columns = load(FEATURES_PATH)

# Load landing-page HTML once at startup
INDEX_HTML = INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.route("/", methods=["GET"])
def index():
    """Styled landing page for the API."""
    return INDEX_HTML


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Body JSON example:

    {
      "age_years": 52.0,
      "gender": 2,
      "height": 170,
      "weight": 80.0,
      "ap_hi": 130,
      "ap_lo": 80,
      "cholesterol": 2,
      "gluc": 1,
      "smoke": 0,
      "alco": 0,
      "active": 1
    }
    """
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON body provided"}), 400

    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(df)[0, 1]
    pred = int(model.predict(df)[0])

    return jsonify(
        {
            "cardio_prediction": pred,
            "cardio_probability": float(proba),
        }
    )


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=9696, debug=True)
