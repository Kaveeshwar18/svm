import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="🔢 AI Digit Predictor",
    page_icon="🔢",
    layout="centered"
)

# ---------------- WORKING CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
    background-size: 400% 400%;
}

/* Card container */
.block-container {
    background: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 20px;
}

/* Title */
h1 {
    text-align: center;
    color: #6a11cb;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

/* Success box */
.result-box {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.title("🔢 AI Handwritten Digit Classifier")
st.write("Upload a PNG image of a handwritten digit (0–9)")

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")

# ---------------- UPLOAD ---------------- #
uploaded_file = st.file_uploader("📤 Upload PNG Image", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((8, 8))
    img = np.array(image)

    img = 255 - img
    img = (img / 255.0) * 16
    img = img.flatten().reshape(1, -1)

    if st.button("🚀 Predict Digit"):
        prediction = model.predict(img)

        st.markdown(
            f"<div class='result-box'>Predicted Digit: {prediction[0]}</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Built with ❤️ by Kaveeshwar")