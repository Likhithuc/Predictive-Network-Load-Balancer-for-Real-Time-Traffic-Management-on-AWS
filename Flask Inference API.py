# realtime_inference_api.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load the model (assume you already copied it from S3 to the same directory)
MODEL_PATH = "smart_scaling_model.h5"
model = load_model(MODEL_PATH)

# Sample: expected number of features = 10 (adjust as per your dataset)
TIME_STEPS = 20
NUM_FEATURES = 10  # set this to your actual input feature size

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['metrics']  # expecting a 2D list [[...], [...], ...]
        if len(data) != TIME_STEPS or len(data[0]) != NUM_FEATURES:
            return jsonify({"error": "Input shape mismatch"}), 400
        
        input_array = np.array(data).reshape(1, TIME_STEPS, NUM_FEATURES)
        prediction = model.predict(input_array)
        scale_out = int(prediction[0][0] > 0.5)

        return jsonify({
            "prediction": float(prediction[0][0]),
            "scale_decision": "scale_out" if scale_out else "no_action"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
