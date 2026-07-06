import numpy as np
import tensorflow as tf

def load_trained_model(model_path):
    """
    Load a trained Keras model.
    """
    return tf.keras.models.load_model(model_path)


def preprocess_image(image):
    """
    Normalize and reshape image for prediction.
    """
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_digit(model, image):
    """
    Predict digit and confidence.
    """
    prediction = model.predict(image, verbose=0)

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return digit, confidence