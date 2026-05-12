# app.py

import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("bow_vectorizer.pkl", "rb"))
emotion_mapping = pickle.load(open("emotion_mapping.pkl", "rb"))

# Reverse Mapping
reverse_mapping = {v: k for k, v in emotion_mapping.items()}

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: #00ADB5;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #B0B0B0;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
    background: linear-gradient(135deg, #1F2937, #111827);
    border: 1px solid #374151;
}

.emotion {
    font-size: 38px;
    font-weight: bold;
    color: #00FFD1;
}

textarea {
    font-size: 18px !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #00ADB5, #007BFF);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3em;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #007BFF, #00ADB5);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown('<div class="title">🧠 Emotion Detection AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect emotions from text using Machine Learning & NLP</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------

user_input = st.text_area(
    "Enter your text below:",
    height=180,
    placeholder="Type something like: I am feeling amazing today..."
)

# ---------------- EMOJI MAP ----------------

emoji_dict = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "happy": "😁",
    "neutral": "😐"
}

# ---------------- PREDICTION ----------------

if st.button("Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    
    else:

        # Vectorize
        vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(vector)[0]

        # Get emotion label
        emotion = reverse_mapping[prediction]

        # Emoji
        emoji = emoji_dict.get(emotion.lower(), "🧠")

        # Display Result
        st.markdown(f"""
        <div class="result-box">
            <div class="emotion">{emoji} {emotion.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <center>
    <p style='color:gray;'>
    Built with ❤️ using Streamlit, NLP & Logistic Regression
    </p>
    </center>
    """,
    unsafe_allow_html=True
)