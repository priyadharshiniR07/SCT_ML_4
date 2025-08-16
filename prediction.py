import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Config
IMG_SIZE = 64
ROI_START = (100, 100)
ROI_END = (300, 300)

# Load model and label encoder
model = load_model("gesture_model.h5")  # or 'gesture_model.h5' if you're using the old format
label_encoder = joblib.load("label_encoder.pkl")
print("[INFO] Model and label encoder loaded successfully.")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("[INFO] Starting real-time gesture recognition. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not received.")
        break

    # Flip and extract Region of Interest
    frame = cv2.flip(frame, 1)
    roi = frame[ROI_START[1]:ROI_END[1], ROI_START[0]:ROI_END[0]]

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm = resized.astype("float32") / 255.0
    norm = norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    prediction = model.predict(norm)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    # Display
    cv2.rectangle(frame, ROI_START, ROI_END, (0, 255, 0), 2)
    cv2.putText(frame, f'Gesture: {predicted_label}', (ROI_START[0], ROI_START[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Leap Gesture Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
