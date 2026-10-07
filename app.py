import streamlit as st
from fastai.vision.all import *
from PIL import Image
import os
import requests
import shutil

st.title("🫁 X-Ray Vision Diagnostic Panel")
st.write("Upload a patient's chest X-ray scan below for rapid automated analysis.")

# 🔗 Paste your direct download link below
MODEL_URL = "https://drive.google.com/file/d/1k9U8pCefLNhcqibW7_uJsqBOnRboR8RF/view?usp=sharing"
MODEL_PATH = "pneumonia_resnet34.pkl"

# Automatically stream-download the weights file if it's missing from the app instance
if not os.path.exists(MODEL_PATH):
    with st.spinner("Initializing 98.37% accurate AI diagnostic parameters... Please wait."):
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            st.success("AI framework loaded successfully!")
        except Exception as e:
            st.error(f"Network error loading model weights: {e}")

uploaded_file = st.file_uploader("Choose an X-ray image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Loaded Patient Input Scan', use_container_width=True)
    st.write("Processing matrix array through neural network...")
    
    try:
        # Load the downloaded Fast.ai model weights file
        learn = load_learner(MODEL_PATH)
        pred, pred_idx, probs = learn.predict(img)
        
        st.subheader(f"Diagnostic Assessment: {pred}")
        st.metric(label="System Prediction Confidence Value", value=f"{probs[pred_idx]*100:.2f}%")
    except Exception as e:
        st.error(f"Error executing neural network inference: {e}")
