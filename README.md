## Live Demo  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://coral-reef-health-ai-jogw9eiudfghfg7ncrpcip.streamlit.app/)

# Coral Reef Health Monitoring AI  

An AI-powered image classification project to **monitor the health of coral reefs**.  
It uses a **Convolutional Neural Network (CNN)** to classify underwater coral images into 3 categories:  

- **Healthy Corals**  
- **Bleached Corals**  
- **Dead Corals**  

This system was trained using **TensorFlow/Keras** and deployed with **Streamlit** for a smooth interactive experience.

---

## Features  
Train & evaluate a custom **CNN model** for coral reef health classification  
Upload coral images and get **instant predictions** with confidence scores  
Clean & responsive **Streamlit web app**  
One-click **deployment** on Streamlit Cloud  

---

## Tech Stack  
- **Python 3**  
- **TensorFlow / Keras** (Model Training & Inference)  
- **Streamlit** (Frontend Web App)  
- **NumPy, Pillow, Matplotlib** (Image processing & visualization)  

---

## Model Architecture  
- **Conv2D** (32 filters, 3×3, ReLU)  
- **MaxPooling2D** (2×2)  
- **Conv2D** (64 filters, 3×3, ReLU)  
- **MaxPooling2D** (2×2)  
- **Conv2D** (128 filters, 3×3, ReLU)  
- **MaxPooling2D** (2×2)  
- **Flatten**  
- **Dense** (128, ReLU)  
- **Dropout** (0.3)  
- **Dense** (3, Softmax)  

**Total Trainable Parameters:** ~3.3 Million  

---

## Example Output  

| Uploaded Image | Model Prediction |
|---------------|----------------|
| ![example coral](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Coral_Reef.jpg/320px-Coral_Reef.jpg) | **Healthy Corals** (92.5% sure) |

---

## Future Improvements  
- Add support for **real-time webcam input**  
- Train with **larger dataset** for better accuracy  
- Deploy as **mobile app** for field researchers  

---

## License  
MIT License – free to use & modify.
