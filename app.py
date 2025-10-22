from flask import Flask, jsonify
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# -------------------------
# Supabase Configuration
# -------------------------
SUPABASE_URL = "https://jawdhtalovhqoorwfrkt.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imphd2RodGFsb3ZocW9vcndmcmt0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA4ODUzMjEsImV4cCI6MjA3NjQ2MTMyMX0.iKrhE_b3lL0CBEQFnTkFVbvK04aqrQ8eWQeloyyMJpg"

SENSOR_URL = f"{SUPABASE_URL}/rest/v1/sensor_data"
PREDICTION_URL = f"{SUPABASE_URL}/rest/v1/predictions"

headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------
# Load Trained Model
# -------------------------
model = joblib.load("savehydro_tds_model.pkl")

# -------------------------
# Helper Functions
# -------------------------
def fetch_latest_sensor_data():
    """Fetch the latest reading from Supabase sensor_data"""
    params = {"select": "*", "order": "recorded_at.desc", "limit": 1}
    res = requests.get(SENSOR_URL, headers=headers, params=params)
    if res.status_code != 200:
        print("❌ Error fetching sensor data:", res.text)
        return None
    data = res.json()
    return data[0] if data else None


def classify_water_quality(tds_value):
    """Classify water based on TDS thresholds"""
    if tds_value < 500:
        return "Good"
    elif tds_value <= 1000:
        return "Moderate (Non-potable)"
    else:
        return "Poor (Unsafe)"


def save_prediction(sensor_id, predicted_tds, status, confidence):
    """Save prediction result to Supabase predictions table"""
    payload = {
        "sensor_id": sensor_id,
        "predicted_tds": round(float(predicted_tds), 2),
        "predicted_status": status,
        "confidence": confidence,
        "created_at": datetime.utcnow().isoformat()
    }
    res = requests.post(PREDICTION_URL, headers=headers, json=payload)
    if res.status_code not in [200, 201]:
        print("⚠️ Failed to save prediction:", res.text)


# -------------------------
# API Endpoint: Predict
# -------------------------
@app.route("/predict", methods=["GET"])
def predict():
    """Fetch latest sensor data, predict resultant TDS, and store result"""
    data = fetch_latest_sensor_data()
    if not data:
        return jsonify({"error": "No sensor data available"}), 404

    try:
        # Extract latest sensor readings
        ro_tds = float(data.get("tds_value", 0))
        rain_tds = ro_tds * 0.8  # assume partial dilution from rainwater
        ro_temp = float(data.get("temperature", 25))
        rain_temp = ro_temp - 1.5

        # Prepare data for prediction
        X = pd.DataFrame([{
            "ro_tds": ro_tds,
            "rain_tds": rain_tds,
            "ro_temp": ro_temp,
            "rain_temp": rain_temp
        }])

        # Make predictions using model ensemble
        preds = [tree.predict(X)[0] for tree in model.estimators_]
        predicted_tds = np.mean(preds)
        std_dev = np.std(preds)
        confidence = round(max(0, 100 - std_dev), 2)
        status = classify_water_quality(predicted_tds)

        # Save to Supabase
        save_prediction(data["id"], predicted_tds, status, confidence)

        return jsonify({
            "sensor_id": data["id"],
            "predicted_tds": round(float(predicted_tds), 2),
            "status": status,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "🌊 SaveHydro ML Prediction API – Auto Deployed Every 30 Minutes"


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
