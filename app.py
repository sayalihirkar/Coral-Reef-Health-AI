import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# ---- CONFIG ----
IMG_SIZE = (128, 128)
class_names = ["bleached_corals", "dead_corals", "healthy_corals"] 

# ---- Load Model ----
model = load_model("CoralModel.keras")

st.title("Coral Reef Health Monitoring AI")
st.write("Upload an underwater coral image and the AI will predict its health status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---- Preprocess ----
    img_resized = img.resize(IMG_SIZE)
    arr = image.img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, 0)

    # ---- Predict ----
    probs = model.predict(arr, verbose=0)[0]
    best = np.argmax(probs)

    st.subheader(f"✅ Final Prediction: **{class_names[best]}** ({probs[best]*100:.2f}% sure)")

    # ---- Show Probabilities (Bar Chart) ----
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(class_names, probs)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig)
