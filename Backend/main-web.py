from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    'http://localhost',
    'http://localhost:3000',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL = tf.keras.models.load_model('../saved_models/2')
CLASS_NAMES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']


def read_image(contents) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_image(await file.read())
    img_resized = cv2.resize(image, dsize=(256, 256))
    img_batch = np.expand_dims(img_resized, axis=0)

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
