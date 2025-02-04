from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import uvicorn

app = FastAPI()

# Serve static files (HTML, CSS, JS)

# Load the trained model
model = tf.keras.models.load_model("C:/Users/VIKRAM RAAJA K/OneDrive/Desktop/Fast/CNN/model/mnist_cnn.h5")

class ImageData(BaseModel):
    image: str

def preprocess_image(image_base64):
    """Convert base64 image to a format suitable for the CNN model."""
    image_data = base64.b64decode(image_base64.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST format (28x28)
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input
    return image

@app.get("/")
def home():
    return FileResponse("C:/Users/VIKRAM RAAJA K/OneDrive/Desktop/Fast/CNN/templates/index.html")  # Serve the HTML file

@app.post("/predict")
async def predict_digit(data: ImageData):
    try:
        image = preprocess_image(data.image)
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)
        return {"prediction": int(predicted_digit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    debug = True