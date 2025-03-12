from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from train_model import get_predicted_value, helper, predict_treatment

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# ---------------------------------------------------------------------
# 1. Connect to MongoDB
# ---------------------------------------------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_db"]
patients_collection = db["patients"]

# ---------------------------------------------------------------------
# 2. Load trained model (for verification only; prediction function loads it internally)
# ---------------------------------------------------------------------
model_path = "model/treatment_model.pkl"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully from:", model_path)
except FileNotFoundError:
    model = None
    print(f"Error: Could not find the model file at {model_path}")

# ---------------------------------------------------------------------
# 3. Flask Routes
# ---------------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({"message": "AI Medicine System API is running"})

@app.route("/add_patient", methods=["POST"])
def add_patient():
    data = request.json
    if not data:
        return jsonify({"error": "No patient data provided"}), 400

    patients_collection.insert_one(data)
    return jsonify({"message": "Patient data saved successfully"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received request data:", data)  # Debug print

    if not isinstance(data, dict) or 'features' not in data:
        return jsonify({'error': 'Invalid request. No features provided.'}), 400

    features = data.get('features')
    if not isinstance(features, dict):
        return jsonify({'error': 'Invalid data format. "features" must be a dictionary.'}), 400

    print("Extracted Features:", features)  # Debug print

    try:
        # Call the ML prediction function from train_model.py
        treatment = predict_treatment(features)
        return jsonify({'treatment': treatment})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------
# 4. Run Flask (Development Mode)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
