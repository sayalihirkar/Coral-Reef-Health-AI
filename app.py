import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# ---- CONFIG ----
IMG_SIZE = (128, 128)
class_names = ["Bleached Corals", "Dead Corals", "Healthy Corals"]

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

    # ---- Print Detailed Predictions ----
    st.write(f"### Predictions for: **{uploaded_file.name}**")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name:<15}: {probs[i]*100:.2f}%")
    st.subheader(f"Final Prediction: **{class_names[best]}** ({probs[best]*100:.2f}% sure)")
