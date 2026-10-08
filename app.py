import streamlit as st
import os
import shutil
import gdown
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# 1. UPDATE YOUR MODEL NETWORK IN_FRASTRUCTURE PATHS
MODEL_URL = 'https://drive.google.com/file/d/1AiysqnARtqethhIpZI1yjmad1ABBnrR0/view?usp=sharing'
MODEL_PATH = 'pneumonia_weights.pth'

st.title("X-Ray Vision Diagnostic Panel")

# Function to download the raw weights tensor file from your Google Drive path natively
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading elite 98.50% accuracy model tensors from Google Drive..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as download_error:
                st.error(f"Google Drive transmission friction loop: {download_error}")
                st.stop()
                
    # INSTANTIATE THE IDENTICAL RESNET34 BASE MODEL TEMPLATE
    # This builds the exact neural network layer layout in system memory
    model = models.resnet34(weights=None)
    
    # Alter the final fully-connected linear layer to match your 2 classes (0: Normal, 1: Pneumonia)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    
    # OVERRIDE AND MAP THE RAW TENSORS NATIVELY
    # This loads your high-performance weights straight onto the server's CPU core
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    classifier = download_and_load_model()
except Exception as init_error:
    st.error(f"System initialization anomaly: {init_error}")
    st.stop()

# 2. FILE UPLOADER LOGIC GRID
uploaded_file = st.file_uploader("Upload Patient Chest Radiography Scan Canvas", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Loaded Patient Input Scan", use_column_width=True)
    st.write("Processing matrix array through neural network...")

    # Match the image preprocessing transformations from your high-velocity Kaggle run
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet metrics
    ])
    img_tensor = transform(img).unsqueeze(0)

    try:
        with torch.no_grad():
            # Run inference over the image tensor matrix array
            raw_outputs = classifier(img_tensor)
            probabilities = torch.nn.functional.softmax(raw_outputs[0], dim=0)
            predicted_index = torch.argmax(probabilities).item()

        # CLASSIFICATION DECODING STRINGS
        CLINICAL_MAP = {
            0: "Normal Pulmonary Matrix Localized - Structure Baseline",
            1: "Pulmonary Consolidation Infiltrates / Pathology Traces Detected"
        }
        final_clinical_string = CLINICAL_MAP.get(predicted_index, "Unknown Layout Detected")

        # STREAM EXPLICIT SUCCESS STRINGS AND METRICS TO THE INTERFACE CONSOLE
        st.markdown("---")
        st.subheader(f"📊 Diagnostic Assessment: {final_clinical_string}")
        st.metric(label="System Prediction Confidence Value", value=f"{probabilities[predicted_index].item() * 100:.2f}%")

        # MANDATORY CLINICAL CONSENSUS INFORMATION LAYER
        st.markdown("""
        <div style="background-color:#f9f9f9; padding:12px; border-left:4px solid #ff4b4b; border-radius:4px; margin-top:15px;">
            <h5 style="margin-top:0; color:#333;">⚠️ Medical Consensus Validation Rules</h5>
            <p style="font-size:0.85em; color:#555; margin-bottom:8px;">
                <strong>1. Visual Feature Cross-Check:</strong> Describe all observed visual variables in the media input file (e.g., bilateral inflation clarity vs. opaque patches of consolidation or fluid drag) before verifying output.
            </p>
            <p style="font-size:0.85em; color:#555; margin-bottom:8px;">
                <strong>2. Differential Assessment Options:</strong> This model presents a technical calculation of probability metrics only. The clinical review should weigh at least three distinct possibilities: Normal baseline layout, focal bacterial infiltration markers, or patchy viral/atypical configurations.
            </p>
            <p style="font-size:0.85em; color:#555; margin-bottom:8px;">
                <strong>3. Label Verification Directive:</strong> Make sure to double-check the physical label, patient profile metrics, and independent findings to confirm this automated computer vision output is accurate. This tool presents general educational information only.
            </p>
            <p style="font-size:0.85em; color:#555; margin-zero:0;">
                <strong>4. Optimization Pathways:</strong> Reviewed diagnostic cases require tracking via distinct pathways: targeted antimicrobial consensus protocols, supportive hydration loops, or continuous diagnostic monitoring.
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as inference_error:
        st.error(f"Inference Engine Friction Loop: {inference_error}")


