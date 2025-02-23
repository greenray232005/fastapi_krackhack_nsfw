import tensorflow.lite as tflite
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

# Define class labels
LABELS = ["Drawing", "Hentai", "Neutral", "Porn", "Sexy"]

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    """Preprocesses image to match the model's input shape."""
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))  # Resize to model input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize (if required by the model)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handles image upload and returns classification results."""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_data = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Get prediction results

    # Format response
    response = [{"label": LABELS[i], "confidence": float(output_data[i])} for i in range(len(LABELS))]
    return {"predictions": response}

# Run with: uvicorn main:app --reload
