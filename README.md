# 🎤 Voice Emotion Detection System

An Artificial Intelligence based application that detects **human emotions from voice recordings** using Machine Learning and audio signal processing techniques.

This system analyzes speech signals, extracts audio features, and predicts emotions such as **Happy, Sad, Angry, Neutral, and Fear**.

---

# 🚀 Features

* 🎧 Upload audio file for emotion detection
* 🎤 Live microphone voice recording
* 📊 Emotion confidence visualization graphs
* 📈 Interactive emotion prediction dashboard
* 🔊 Audio waveform visualization
* 📋 Emotion prediction details table
* 🌙 Dark UI professional dashboard design

---

# 🧠 Machine Learning Workflow

Voice Input
⬇
Feature Extraction (MFCC)
⬇
Random Forest Classifier
⬇
Emotion Prediction
⬇
Visualization Dashboard

---

# 🛠 Technologies Used

* **Python**
* **Streamlit**
* **Librosa**
* **Scikit-learn**
* **NumPy**
* **Plotly**

---

# 📂 Project Structure

```
voice-emotion-detection-system
│
├── app.py
├── train_model.py
├── predict_emotion.py
├── feature_extraction.py
├── requirements.txt
├── README.md
│
├── dataset
│   ├── angry
│   ├── happy
│   ├── sad
│   ├── neutral
│   └── fear
│
└── models
    └── emotion_model.pkl
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/yourusername/voice-emotion-detection-system.git
```

Navigate to project folder

```
cd voice-emotion-detection-system
```

Create virtual environment

```
python -m venv venv
```

Activate environment

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run The Application

```
streamlit run app.py
```

Then open browser

```
http://localhost:8501
```

---

# 📊 Model Information

Algorithm Used:

**Random Forest Classifier**

Feature Extraction:

**MFCC (Mel Frequency Cepstral Coefficients)**

Audio Processing:

**Librosa Library**

---

# 🎯 Future Improvements

* Add more emotion classes
* Improve model accuracy with deep learning
* Add real-time emotion detection
* Deploy the system online

---

# 👨‍💻 Author

**Tejesh Gonupalli**

MCA Final Year Project
Voice Emotion Detection System

---

# ⭐ If you like this project

Please give it a ⭐ on GitHub!
