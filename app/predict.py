import os
import numpy as np
import tensorflow as tf

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# -----------------------------
# Load MNIST Test Dataset
# -----------------------------
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# -----------------------------
# Select One Image
# -----------------------------
index = 25

image = x_test[index]

actual_label = y_test[index]

# -----------------------------
# Preprocess
# -----------------------------
image = image.astype("float32") / 255.0

image = np.expand_dims(image, axis=0)

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(image, verbose=0)

predicted_digit = np.argmax(prediction)

confidence = np.max(prediction) * 100

# -----------------------------
# Output
# -----------------------------
print(f"Actual Digit    : {actual_label}")

print(f"Predicted Digit : {predicted_digit}")

print(f"Confidence      : {confidence:.2f}%")