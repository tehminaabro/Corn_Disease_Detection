from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
import numpy as np
import json
from PIL import Image
import io

# FastAPI App
app = FastAPI()

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="Tehmina-Abro/Corn-Disease-Detection-Model",
    filename="corn_disease_model.keras"
)

# Load Model
model = load_model(model_path)

# Load Class Names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.get("/")
def home():
    return {
        "message": "Corn Disease Detection API is Running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {
            "error": "Invalid image file."
        }

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction) * 100)

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }
