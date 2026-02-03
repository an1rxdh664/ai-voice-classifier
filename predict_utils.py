import numpy as np
import joblib
from extract import extract_features

# Load trained model & scaler ONCE (global)
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_from_file(file_path):

    # Extract 20D feature vector
    feat = extract_features(file_path)
    feat_arr = np.array(feat).reshape(1, -1)
    feat_scaled = scaler.transform(feat_arr)

    # Model prediction
    prediction = model.predict(feat_scaled)[0]             # 0 or 1
    probability = model.predict_proba(feat_scaled)[0]      # [prob_human, prob_ai]

    prediction_label = "AI" if prediction == 1 else "Human"
    confidence = float(probability[prediction])                  # prob of chosen label

    return {
        "label": prediction_label,
        "prediction_index": int(prediction),
        "probabilities": {
            "human": float(probability[0]),
            "ai": float(probability[1])
        },
        "confidence": confidence
    }
