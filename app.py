from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
import joblib  # gunakan joblib, bukan pickle

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model_random_forest.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model Random Forest berhasil dimuat!")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise

# Fungsi deskripsi hasil prediksi
def get_health_description(pred_class):
    descriptions = {
        "BAIK": "Kualitas udara baik dan tidak memiliki risiko bagi kesehatan.",
        "SEDANG": "Kualitas udara sedang. Beberapa polutan mungkin menyebabkan efek ringan.",
        "TIDAK SEHAT": "Kualitas udara tidak sehat. Anggota kelompok sensitif mungkin mengalami efek kesehatan.",
    }
    return descriptions.get(pred_class, "Deskripsi tidak tersedia")

@app.route('/')
def home():
    return render_template('index.html')

import pandas as pd
# ...existing code...

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Urutan fitur harus sama dengan saat training!
        feature_names = [
            'pm_sepuluh',
            'pm_duakomalima',
            'sulfur_dioksida',
            'karbon_monoksida',
            'ozon',
            'nitrogen_dioksida'
        ]
        features = [[
            float(data['pm_sepuluh']),
            float(data['pm_duakomalima']),
            float(data['sulfur_dioksida']),
            float(data['karbon_monoksida']),
            float(data['ozon']),
            float(data['nitrogen_dioksida'])
        ]]
        features_df = pd.DataFrame(features, columns=feature_names)
        prediction = model.predict(features_df)
        pred_class = prediction[0]

        return jsonify({
            'prediction': pred_class,
            'health_status': pred_class,
            'description': get_health_description(pred_class)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'model_type': type(model).__name__,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True)