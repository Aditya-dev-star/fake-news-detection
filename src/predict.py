import joblib
import numpy as np
from src.preprocess import clean_text

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    decision = model.decision_function(vector)[0]

    confidence = 1 / (1 + np.exp(-abs(decision)))
    confidence = round(confidence * 100, 2)

    label = "REAL 📰" if decision > 0 else "FAKE 🚫"

    return label, confidence
