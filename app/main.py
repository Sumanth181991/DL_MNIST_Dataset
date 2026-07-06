import io
import os

from PIL import Image

from fastapi import FastAPI, UploadFile, File

from app.utils import (
    load_trained_model,
    preprocess_image,
    predict_digit
)

app = FastAPI(
    title="MNIST ANN API",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_model.keras"
)

model = load_trained_model(MODEL_PATH)


@app.get("/")
def home():

    return {
        "message": "MNIST ANN API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    image = preprocess_image(image)

    digit, confidence = predict_digit(
        model,
        image
    )

    return {
        "predicted_digit": digit,
        "confidence": f"{confidence:.2f}%"
    }