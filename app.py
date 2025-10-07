import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import joblib
import requests
import zipfile

# --------------------------
# Google Drive download helper
# --------------------------
def download_from_drive(file_id, dest_path):
    """Bypass Google Drive virus-scan confirmation"""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    return dest_path

# --------------------------
# Model download, extraction, and loading (safe version)
# --------------------------
# --------------------------  
# Model download, loading (no zip)  
# --------------------------

os.makedirs("models", exist_ok=True)

file_ids = {
    "unet_model.h5": "1op3ApQ50GOvH_p3CFRkvQfxJMOEywcSu",
    "ensemble_model.pkl": "1mEpPNckyAS7Ud6enp5LEq7bDSQVHJVl7",
    "label_encoder.pkl": "13hCEDrj8gX0jkemVEkL1LwN3g4BZOJFV"
}

# Download models if missing
for fname, fid in file_ids.items():
    path = os.path.join("models", fname)
    if not os.path.exists(path):
        st.info(f"📥 Downloading {fname} ... please wait")
        download_from_drive(fid, path)
        st.success(f"✅ {fname} downloaded successfully.")

# Load U-Net model
model_path = "models/unet_model.h5"
if not os.path.exists(model_path):
    st.error("❌ U-Net model file not found!")
    st.stop()
else:
    model = load_model(model_path)
    st.success("✅ U-Net model loaded successfully!")

# Load ensemble model + encoder
try:
    ensemble_model = joblib.load("models/ensemble_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    ensemble_available = True
except Exception as e:
    st.warning(f"⚠️ Could not load ensemble: {e}")
    ensemble_available = False

# --------------------------
# Global Settings
# --------------------------
IMG_SIZE = 128

# --------------------------
# Preprocessing
# --------------------------
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    norm_img = img / 255.0
    return img, np.expand_dims(norm_img, axis=(0, -1))

# --------------------------
# KBIS Rule-based Reasoning
# --------------------------
def classify_blockage(mask):
    thresholds = [0.5, 0.3, 0.25]
    lumen_area_raw = 0
    chosen_thr = 0.5

    for thr in thresholds:
        binary_mask = (mask > thr).astype("uint8")
        area = np.sum(binary_mask)
        if area > 200:
            lumen_area_raw = area
            chosen_thr = thr
            break

    if lumen_area_raw == 0:
        lumen_area_raw = np.sum(mask > 0.1)
        chosen_thr = 0.1

    if lumen_area_raw < 1000:
        correction_factor = 20
    elif lumen_area_raw < 3000:
        correction_factor = 30
    else:
        correction_factor = 40

    lumen_area = lumen_area_raw * correction_factor
    severe_thr = 9000
    normal_thr = 18000

    if lumen_area < severe_thr:
        return ("Severe Blockage",
                f"Raw area={lumen_area_raw}, Scaled={lumen_area:.0f} < {severe_thr} → Severe "
                f"(mask thr {chosen_thr}, ×{correction_factor})",
                "#E74C3C", "🔴")
    elif lumen_area < normal_thr:
        return ("Moderate Blockage",
                f"Raw area={lumen_area_raw}, Scaled={lumen_area:.0f} between {severe_thr}–{normal_thr} → Moderate "
                f"(mask thr {chosen_thr}, ×{correction_factor})",
                "#F1C40F", "🟡")
    else:
        return ("Normal",
                f"Raw area={lumen_area_raw}, Scaled={lumen_area:.0f} ≥ {normal_thr} → Normal "
                f"(mask thr {chosen_thr}, ×{correction_factor})",
                "#2ECC71", "🟢")

# --------------------------
# Feature Extraction for Ensemble
# --------------------------
def extract_features(mask):
    binary_mask = (mask > 0.5).astype("uint8")
    vessel_ratio = np.sum(binary_mask) / mask.size
    avg_intensity = np.mean(mask[binary_mask == 1]) if np.sum(binary_mask) > 0 else 0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0

    return np.array([[vessel_ratio, avg_intensity, perimeter]])

# --------------------------
# Streamlit UI
# --------------------------
st.markdown("""
    <style>
    .main { background-color: #121212; color: #EAECEE; }
    body { background-color: #121212; color: #EAECEE; }
    h1, h2, h3, h4 {
        color: #1ABC9C;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stFileUploader label {
        color: #F1C40F !important;
        font-weight: bold;
    }
    .stButton button {
        background-color: #1ABC9C;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
    }
    .stButton button:hover {
        background-color: #16A085;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 Blood Vessel Blockage Detection")
st.write("Select a mode below to analyze the uploaded ultrasound image:")

mode = st.radio("Select Mode:", ["KBIS (Rule-Based Reasoning)", "AML (Ensemble Model)"])

uploaded_file = st.file_uploader("📤 Upload Ultrasound Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    orig_img, input_img = preprocess_image(uploaded_file)
    pred_mask = model.predict(input_img)[0, :, :, 0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(orig_img, channels="GRAY", use_container_width=True)
    with col2:
        st.subheader("Predicted Segmentation")
        st.image((pred_mask > 0.5).astype("uint8") * 255, channels="GRAY", use_container_width=True)

    if mode == "KBIS (Rule-Based Reasoning)":
        diagnosis, reasoning, color, emoji = classify_blockage(pred_mask)

        st.markdown(f"""
        <div style="padding:20px; border-radius:12px;
                    background:linear-gradient(135deg, {color}, #2C3E50);
                    text-align:center; color:white; margin-top:20px;
                    box-shadow: 0px 4px 15px rgba(0,0,0,0.7);">
            <h2>{emoji} {diagnosis}</h2>
            <p style="font-size:16px;">{reasoning}</p>
        </div>
        """, unsafe_allow_html=True)

    elif mode == "AML (Ensemble Model)":
        if not ensemble_available:
            st.error("⚠️ Ensemble model not found! Please check model links or files.")
        else:
            features = extract_features(pred_mask)
            pred_class = ensemble_model.predict(features)[0]
            prob = ensemble_model.predict_proba(features)[0]
            confidence = np.max(prob) * 100
            class_name = label_encoder.inverse_transform([pred_class])[0]

            st.markdown(f"""
            <div style="padding:20px; border-radius:12px;
                        background:linear-gradient(135deg, #3498DB, #2C3E50);
                        text-align:center; color:white; margin-top:20px;
                        box-shadow: 0px 4px 15px rgba(0,0,0,0.7);">
                <h2>🤖 AML Prediction: {class_name}</h2>
                <p style="font-size:16px;">Confidence: {confidence:.2f}%</p>
                <p style="font-size:14px;">Features → Vessel Ratio: {features[0][0]:.4f}, Intensity: {features[0][1]:.4f}, Perimeter: {features[0][2]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
---
💡 <span style="color:#1ABC9C;">Powered by U-Net + KBIS + Ensemble Learning (AML)</span> | © 2025 CISKA Research
""", unsafe_allow_html=True)







