import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# CONFIG
DATASET_PATH = r"C:\Users\Priyadharshini\Downloads\archive\leapGestRecog"
IMG_SIZE = 64

# 1. Load dataset
def load_dataset():
    X, y = [], []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                label = os.path.basename(root).lower()  # folder name is label
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

print("[INFO] Loading dataset...")
X, y = load_dataset()

# Debugging info
print(f"[DEBUG] Total images loaded: {len(X)}")
print(f"[DEBUG] Total labels loaded: {len(y)}")
if len(y) == 0:
    raise ValueError("No labels found! Check DATASET_PATH and folder structure.")

# 2. Preprocessing
X = X.astype("float32") / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# 3. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")
print("[INFO] Saved label encoder with classes:", list(label_encoder.classes_))

# 4. Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# 5. Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 6. Train model
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=[early_stop]
)

# 7. Save model
model.save("gesture_model.h5")
print("[INFO] Model saved as gesture_model.h5")

# 8. Plot training history
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("training_accuracy.png")
plt.show()
