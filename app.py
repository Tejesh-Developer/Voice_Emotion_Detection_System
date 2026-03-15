import streamlit as st
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sounddevice as sd
import soundfile as sf

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Voice Emotion Detection",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f2027,#203a43,#2c5364);
}

section[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#141e30,#243b55);
    padding-top:20px;
}

section[data-testid="stSidebar"] h2 {
    color:white;
    font-size:22px;
}

section[data-testid="stSidebar"] label {
    color:#e5e7eb;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)
# -------------------------------
# HEADER
# -------------------------------

st.markdown("""
        <style>
        .extract-box {
            background-color: #161B22;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 🔥 HERO BANNER
st.markdown("""
    <div style="
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    padding:35px;
    border-radius:20px;
    margin-bottom:25px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.4);
    ">

    <h1 style="
    color:white;
    font-size:42px;
    font-weight:700;
    margin-bottom:10px;">
    🎙 Voice Emotion Detection System
    </h1>

    <p style="
    color:#cbd5e1;
    font-size:18px;
    margin:0;">
    AI system that detects human emotions from voice
    </p>

    </div>
    """, unsafe_allow_html=True)
# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.markdown("""
<div style="
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
padding:15px;
border-radius:10px;
text-align:center;
margin-bottom:20px;
">

<h3 style="color:white;margin:0;">🎙 Emotion AI</h3>
<p style="color:#cbd5e1;font-size:13px;margin:0;">
Voice Analysis Dashboard
</p>

</div>
""", unsafe_allow_html=True)



st.sidebar.markdown("### 📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Upload Audio", "Model Performance", "About Project"]
)

# if theme:
#     # Dark Theme
#     st.markdown("""
#     <style>
#     .stApp {
#         background-color:#0f172a;
#         color:white;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# else:
#     # Light Theme
#     st.markdown("""
#     <style>
#     .stApp {
#         background-color:#f5f5f5;
#         color:black;
#     }
#     </style>
#     """, unsafe_allow_html=True)
# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("models/emotion_model.pkl")

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------

def extract_features(file):

    audio, sr = librosa.load(file, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)

    return mfcc_scaled


# -------------------------------
# PREDICT EMOTION
# -------------------------------

def predict_emotion(file):

    features = extract_features(file)

    prediction = model.predict([features])

    probs = model.predict_proba([features])[0]

    return prediction[0], probs

def show_prediction_results(audio_path, emotion, probs):

    import librosa.display

    # waveform
    audio, sr = librosa.load(audio_path)

    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#1e3c72,#2a5298);
        padding:20px;
        border-radius:15px;
        margin-top:25px;
        margin-bottom:15px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.3);
        ">

        <h3 style="
        color:white;
        font-size:22px;
        margin-bottom:5px;">
        🎧 Voice Waveform
        </h3>

        <p style="
        color:#cbd5e1;
        font-size:14px;
        margin:0;">
        Audio signal visualization of the recorded/uploaded voice
        </p>

        </div>
        """, unsafe_allow_html=True)

    fig, ax = plt.subplots()

    librosa.display.waveshow(audio, sr=sr, ax=ax)

    ax.set_title("Audio Waveform")

    st.pyplot(fig)

    # emotion card
    emoji_map = {
        "happy":"😊",
        "sad":"😢",
        "angry":"😡",
        "fear":"😨",
        "neutral":"😐"
    }

    emoji = emoji_map.get(emotion.lower(),"🎤")

    st.markdown(f"""
    <div style="
    padding:25px;
    border-radius:12px;
    background:#111827;
    color:white;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    border:2px solid #374151;">
    {emoji} Detected Emotion: {emotion.upper()}
    </div>
    """, unsafe_allow_html=True)

    confidence = max(probs)

    # gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence*100,
        title={'text': "Prediction Confidence"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#22c55e"},
            'bgcolor': "black"
        }
    ))

    st.plotly_chart(fig, width="stretch")

    # bar chart
    emotions = model.classes_

    df = pd.DataFrame({
        "Emotion": emotions,
        "Confidence": probs
    })

    fig = px.bar(
        df,
        x="Emotion",
        y="Confidence",
        color="Emotion",
        text="Confidence"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, width="stretch")

    # table
    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#283c86,#45a247);
        padding:20px;
        border-radius:15px;
        margin-top:25px;
        margin-bottom:15px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.3);
        ">

        <h3 style="
        color:white;
        font-size:22px;
        margin-bottom:5px;">
        📊 Prediction Details
        </h3>

        <p style="
        color:#e5e7eb;
        font-size:14px;
        margin:0;">
        Emotion confidence values predicted by the AI model
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.dataframe(df, width="stretch")



def record_audio(duration=3, fs=44100):

    st.info("🎤 Recording... Speak now")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    file_path = "recorded_audio.wav"

    sf.write(file_path, recording, fs)

    st.success("Recording complete!")

    return file_path

# -------------------------------
# PAGE 1 : UPLOAD AUDIO
# -------------------------------

if page == "Upload Audio":

    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#1f4037,#99f2c8);
        padding:25px;
        border-radius:18px;
        margin-bottom:15px;
        ">

        <h2 style="color:white;font-size:28px;">
        📂 Upload Voice File
        </h2>

        <p style="color:#e5e7eb;font-size:15px;">
        Upload a WAV audio file to detect emotion
        </p>

        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload WAV file",
        type=["wav"]
    )
    
    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#141e30,#243b55);
        padding:20px;
        border-radius:15px;
        margin-top:20px;
        margin-bottom:15px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.3);
        ">

        <h3 style="
        color:white;
        font-size:22px;
        margin-bottom:5px;">
        🎤 Record Voice
        </h3>

        <p style="
        color:#cbd5e1;
        font-size:14px;
        margin:0;">
        Record your voice using microphone to detect emotion
        </p>

        </div>
        """, unsafe_allow_html=True)

    if st.button("Start Recording", key="record_btn"):

        audio_file = record_audio()

        st.audio(audio_file)

        # save recording
        st.session_state["recorded_audio"] = audio_file
        
    # show detect button after recording
    if "recorded_audio" in st.session_state:

        if st.button("🔍 Detect Emotion from Recording"):

            emotion, probs = predict_emotion(st.session_state["recorded_audio"])

            st.session_state["record_result"] = (emotion, probs)


        if "record_result" in st.session_state:

            emotion, probs = st.session_state["record_result"]

            show_prediction_results(st.session_state["recorded_audio"], emotion, probs)

    if uploaded_file:

        st.audio(uploaded_file)

        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        if st.button("🔍 Detect Emotion", key="detect_btn"):

            emotion, probs = predict_emotion("temp.wav")

            st.session_state["upload_result"] = (emotion, probs)


        if "upload_result" in st.session_state:

            emotion, probs = st.session_state["upload_result"]

            show_prediction_results("temp.wav", emotion, probs)


# -------------------------------
# PAGE 2 : MODEL PERFORMANCE
# -------------------------------

elif page == "Model Performance":

    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#1f4037,#99f2c8);
        padding:25px;
        border-radius:20px;
        margin-bottom:20px;
        ">

        <h2 style="color:white;margin-bottom:5px;">
        📊 Model Performance
        </h2>

        <p style="color:#e5e7eb;font-size:14px;margin:0;">
        Evaluation metrics of the trained emotion detection model
        </p>

        </div>
        """, unsafe_allow_html=True)
    
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=86,
        title={'text': "Model Accuracy"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range':[0,50],'color':'#ef4444'},
                {'range':[50,75],'color':'#facc15'},
                {'range':[75,100],'color':'#22c55e'}
            ]
        }
    ))

    st.plotly_chart(fig, width="stretch")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Accuracy", "86%", "Good")

    with col2:
        st.metric("Total Emotions", "5")

    with col3:
        st.metric("Dataset Samples", "200+")

    st.write("")

    st.info("""
    **Model Used:** Random Forest Classifier  
    **Feature Extraction:** MFCC (Mel Frequency Cepstral Coefficients)  
    **Dataset:** RAVDESS Emotional Speech Dataset
    """)


# -------------------------------
# PAGE 3 : ABOUT PROJECT
# -------------------------------

elif page == "About Project":

    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#141E30,#243B55);
        padding:25px;
        border-radius:20px;
        margin-bottom:20px;
        ">

        <h2 style="color:white;">ℹ About Project</h2>

        <p style="color:#cbd5e1;font-size:15px;">
        Voice Emotion Detection System is an Artificial Intelligence based application
        that analyzes speech signals and predicts the emotional state of a person
        using machine learning techniques.
        </p>

        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="
        background:#1e293b;
        padding:20px;
        border-radius:15px;
        margin-bottom:20px;
        ">

        <h3 style="color:white;">🛠 Technologies Used</h3>

        <ul style="color:#cbd5e1;">
        <li>Python</li>
        <li>Streamlit</li>
        <li>Librosa</li>
        <li>Scikit-learn</li>
        <li>NumPy</li>
        <li>Plotly</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="
        background: linear-gradient(135deg,#134E5E,#71B280);
        padding:20px;
        border-radius:15px;
        ">

        <h3 style="color:white;">🚀 Project Features</h3>

        <ul style="color:#ecfdf5;">
        <li>Upload audio file for emotion detection</li>
        <li>Live microphone voice recording</li>
        <li>Audio waveform visualization</li>
        <li>Emotion confidence graph</li>
        <li>Prediction confidence gauge</li>
        <li>Interactive dashboard UI</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)
