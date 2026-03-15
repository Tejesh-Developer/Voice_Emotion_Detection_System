import joblib
from utils.feature_extraction import extract_features

model = joblib.load("models/emotion_model.pkl")

def predict(file):

    features = extract_features(file)

    emotion = model.predict([features])

    return emotion[0]