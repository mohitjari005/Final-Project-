import warnings
import streamlit as st
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


st.set_page_config(page_title="Eye Image Analyzer", layout="centered")

# -----------------------------
# Dark Mode Toggle
# -----------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# -----------------------------
# Custom Theme Styling
# -----------------------------
if dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .stTextInput > div > div > input {
            color: white;
        }
        .stButton button {
            background-color: #1f77b4;
            color: white;
        }
        .stMetric {
            background-color: #1c1c1c !important;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stMetric {
            background-color: #f0f0f0 !important;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_retina_check_model():
    return load_model("binary_retina_classifier.keras")

@st.cache_resource
def load_disease_model():
    return load_model("Cnn_model.h5")

retina_model = load_retina_check_model()
disease_model = load_disease_model()

# -----------------------------
# Disease Class Labels
# -----------------------------
class_names = ['normal', 'cataract', 'glaucoma', 'diabetic_retinopathy']

# -----------------------------
# Groq API for Medical Info
# -----------------------------
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_disease_info(disease_name):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": f"Overview of {disease_name} in simple terms, symptoms and treatment."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        return f"Error: {res.status_code}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# -----------------------------
# UI Title
# -----------------------------
st.markdown(
    f"<h1 style='text-align: center; color: {'#FAFAFA' if dark_mode else '#1c1c1c'};'>👁️ Eye Image Analyzer</h1>",
    unsafe_allow_html=True
)
st.write("Upload a retina image for verification and disease classification using AI.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='🖼️ Uploaded Image',width=300)

    img_size = (256, 256)
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Retina Verification
    with st.spinner("🔍 Verifying retina image..."):
        retina_pred = retina_model.predict(img_array)[0][0]
        is_retina = retina_pred > 0.5

    if not is_retina:
        st.error("🚫 Not a valid retinal image. Please upload a proper retina scan.")
    else:
        st.success("✅ Retina image verified.")

        # Disease Prediction
        with st.spinner("🧠 Analyzing image for eye disease..."):
            disease_pred = disease_model.predict(img_array)
            if disease_pred.shape[1] == 1:
                pred_class = int(disease_pred[0][0] > 0.5)
                confidence = disease_pred[0][0] if pred_class else 1 - disease_pred[0][0]
            else:
                pred_class = np.argmax(disease_pred)
                confidence = np.max(disease_pred)

        predicted_condition = class_names[pred_class]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏷️ Predicted Condition", predicted_condition.replace("_", " ").title())
        with col2:
            st.metric("📊 Confidence", f"{confidence:.1%}")

        if predicted_condition != "normal":
            st.markdown("---")
            st.subheader("📋 Medical Info")
            with st.spinner("Fetching medical explanation..."):
                info = get_disease_info(predicted_condition)
            st.markdown(f"<div style='color: {'#FAFAFA' if dark_mode else '#000000'}'>{info}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: {'#ff4b4b'}'><b>⚠️ This is not a medical diagnosis. Always consult a doctor.</b></div>", unsafe_allow_html=True)
        else:
            st.success("🎉 Eye appears healthy. Keep up with regular checkups!")

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("This tool verifies if an image is a retina scan and classifies eye diseases using AI.")

