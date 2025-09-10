## Live Demo  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://coral-reef-health-ai-jogw9eiudfghfg7ncrpcip.streamlit.app/)

# Coral Reef Health Monitoring AI  

An AI-powered image classification project to monitor the health of coral reefs.  
This project uses **Convolutional Neural Networks (CNNs)** to classify underwater coral images into 3 categories:  

- Healthy Corals  
- Bleached Corals  
- Dead Corals  

The system is built with **TensorFlow/Keras** for model training and **Streamlit** for an interactive web application.  


# Features  
- Train & evaluate a **CNN model** for coral reef health classification.  
- Upload coral reef images and get instant predictions with probability.  
- User-friendly **Streamlit web app** for demonstrations.  
- Deployable on **Streamlit Cloud** so anyone can access online.  


# Tech Stack  
- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy, Pillow, Matplotlib  

# Model Architecture
- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Conv2D (128 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Flatten
- Dense (128, ReLU)
- Dropout (0.3)
- Dense (3, Softmax)

Total Parameters: 3.3 Million
