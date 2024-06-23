# app.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
import onnxruntime as ort

app = FastAPI()

# Load ONNX model
ort_session = ort.InferenceSession("mnist_cnn.onnx")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image, dtype=np.float32)
    image = (image / 255.0) - 0.5
    image = image[np.newaxis, np.newaxis, :, :]
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    input_data = preprocess_image(image)

    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs[0])

    return {"prediction": int(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
