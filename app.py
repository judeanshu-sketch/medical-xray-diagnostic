import streamlit as st
from fastai.vision.all import *
from PIL import Image

st.title("🫁 X-Ray Vision Diagnostic Panel")
st.write("Upload a patient's chest X-ray scan below for rapid automated analysis.")

uploaded_file = st.file_uploader("Choose an X-ray image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Display the uploaded image cleanly to the user
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Loaded Patient Input Scan', use_container_width=True)
    st.write("Processing matrix array through neural network...")
    
    # 2. Safely load your 98.37% accurate Fast.ai model file
    learn = load_learner('pneumonia_resnet34.pkl')
    
    # 3. Run the image through the network weights
    pred, pred_idx, probs = learn.predict(img)
    
    # 4. Render the diagnostic outputs beautifully
    st.subheader(f"Diagnostic Assessment: {pred}")
    st.metric(label="System Prediction Confidence Value", value=f"{probs[pred_idx]*100:.2f}%")