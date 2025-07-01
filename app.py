import streamlit as st
import numpy as np
from PIL import Image
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load both models
@st.cache_resource
def load_retina_check_model():
    return load_model("/mnt/data/binary_retina_classifier.keras")

@st.cache_resource
def load_disease_model():
    return load_model("Cnn_model.h5")

retina_model = load_retina_check_model()
disease_model = load_disease_model()

# Set class names
class_names = ['normal', 'cataract', 'glaucoma', 'diabetic_retinopathy']

# Groq API setup (optional)
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_disease_info(disease_name):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a medical assistant..."},
            {"role": "user", "content": f"Overview of {disease_name}..."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        return f"Error: {res.status_code}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# Streamlit UI
st.title("👁️ Eye Image Analysis")
st.write("Upload an eye image. The app will first verify it's a retina image, then classify any disease.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess
    img_size = (256, 256)
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Step 1: Retina image verification
    with st.spinner("🔎 Checking if this is a retina image..."):
        retina_pred = retina_model.predict(img_array)[0][0]
        is_retina = retina_pred > 0.5

    if not is_retina:
        st.error("🚫 This image does not appear to be a valid retinal image. Please upload a proper retina scan.")
    else:
        st.success("✅ Retina image confirmed.")
        with st.spinner("🧠 Analyzing for eye disease..."):
            disease_pred = disease_model.predict(img_array)
            if disease_pred.shape[1] == 1:
                pred_class = int(disease_pred[0][0] > 0.5)
                confidence = disease_pred[0][0] if pred_class else 1 - disease_pred[0][0]
            else:
                pred_class = np.argmax(disease_pred)
                confidence = np.max(disease_pred)

        predicted_condition = class_names[pred_class]
        
        # Display result
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏷️ Predicted Condition", predicted_condition.replace("_", " ").title())
        with col2:
            st.metric("📊 Confidence", f"{confidence:.1%}")

        # Step 3: Medical Info if disease
        if predicted_condition != "normal":
            st.markdown("---")
            st.markdown("### 📋 Medical Info")
            with st.spinner("Getting medical info..."):
                info = get_disease_info(predicted_condition)
            st.markdown(info)
            st.warning("⚠️ This is not medical advice. Please consult a doctor.")
        else:
            st.success("🎉 Eye appears to be healthy. Keep up regular checkups!")

# Sidebar info
st.sidebar.markdown("### ℹ️ About")
st.sidebar.write("This tool verifies if an image is a retina scan and classifies diseases using AI.")