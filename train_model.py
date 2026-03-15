import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import extract_features

dataset_path = "dataset"

X = []
y = []

for emotion in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, emotion)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder, file)

        try:
            features = extract_features(file_path)

            X.append(features)
            y.append(emotion)

        except:
            print("Skipping file:", file)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/emotion_model.pkl")