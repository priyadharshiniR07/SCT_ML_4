Hand Gesture Recognition Using Leap Motion (Infrared Images)
This project implements a deep learning-based hand gesture recognition system using grayscale infrared images captured from the Leap Motion controller. It classifies 10 different hand gestures in real-time via a webcam, enabling intuitive human-computer interaction.

🔍 Project Overview
📸 Input: Grayscale infrared images of hands performing gestures
🧠 Model: Convolutional Neural Network (CNN) built using TensorFlow/Keras
🎯 Output: Real-time classification of gestures (e.g., palm, l, fist, etc.)
🎥 Interface: Real-time webcam gesture detection with on-screen prediction
📁 Dataset Info
Dataset: Leap Motion Hand Gesture Dataset (Infrared)
Source: LeapGestRecog (https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
Structure: leapGestRecog/ ├── 00/ │ ├── 01_palm/ │ ├── 02_l/ ├── ... ├── 09/
10 subjects (folders 00–09)
10 different gesture classes (e.g., palm, l, fist, down, etc.)
Format: Grayscale .png images
🛠 Installation
Install all dependencies: pip install tensorflow opencv-python scikit-learn joblib matplotlib

🧠 Model Training
To train the CNN model: python train_gesture_model.py

Saves the model as gesture_model.h5
Saves label encoder as label_encoder.pkl
Outputs training accuracy plot training_accuracy.png
🎥 Real-Time Prediction
To run the webcam-based gesture recognition: python predict.py

Opens webcam feed
Displays predicted gesture label on-screen
Press ESC to quit
🗂 File Structure
. ├── gesture_model.h5 # Trained CNN model (HDF5 format) ├── label_encoder.pkl # Encoded class labels ├── train_gesture_model.py # Script for training the model ├── predict.py # Real-time prediction with webcam ├── training_accuracy.png # Accuracy plot ├── README.txt # Project documentation

📈 Performance
✅ Achieved validation accuracy: 99.96%
📉 No signs of overfitting
🚀 Model generalizes well across subjects


📚 Future Work
Add gesture-based control for media/apps
Enable voice output for recognized gestures
Deploy the model using Flask or Streamlit
