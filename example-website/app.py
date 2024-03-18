# pip install fastapi uvicorn python-multipart
"""
@Author: Canmert Demir
@Date: 2024-03-14
@Email: canmertdemir2@gmail.com
# pip install fastapi uvicorn python-multipart => Fast Api Install

To run app : uvicorn app:app --reload
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.responses import HTMLResponse
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2

app = FastAPI()

json_file = open('trained-model/CNN_three_layer_fully_connected.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('trained-model/CNN_three_fully_connected.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def preprocess_image(image):
    resized_image = cv2.resize(image, (32, 32))
    normalized_image = resized_image.astype('float32') / 255
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        preprocessed_image = preprocess_image(image)

        predictions = loaded_model.predict(preprocessed_image)

        predicted_class_index = np.argmax(predictions)
        predicted_class_label = labels[predicted_class_index]

        return JSONResponse(content={"predicted_class": predicted_class_label}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
async def upload_image_form():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)