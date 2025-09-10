import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Load model (make sure CoralModel.keras is in the same folder or give full path)
model = load_model("CoralModel.keras")

# Define classes
class_names = ["Bleached Corals", "Dead Corals", "Healthy Corals"]

IMG_SIZE = (128, 128)

st.title("Coral Reef Health Monitoring AI")
st.write("Upload an underwater coral image and the AI will predict its health status.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    # Prediction
    preds = model.predict(arr)
    probs = preds[0]

    # Show all class probabilities (like Colab)
    st.write("### Predictions:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probs[i]*100:.2f}%")

    # Final best prediction
    best = np.argmax(probs)
    st.subheader(f"Final Prediction: {class_names[best]} ({probs[best]*100:.2f}% sure)")
