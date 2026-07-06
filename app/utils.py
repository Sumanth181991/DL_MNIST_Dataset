import numpy as np
import tensorflow as tf
from PIL import ImageOps

from PIL import Image


def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Invert colors
    image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to NumPy array
    image = np.array(image)

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def predict_digit(model, image):

    prediction = model.predict(image, verbose=0)

    digit = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    return digit, confidence