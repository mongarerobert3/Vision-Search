from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io
import base64

# Initialize FastAPI app
app = FastAPI()

# Load pretrained YOLO model
model = YOLO("ultralytics/yolov8n")  # YOLOv8 Nano from Hugging Face

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)  # Run inference
    output_img = results[0].plot()  # Draw bounding boxes
    
    # Convert image to base64 to send as JSON response
    _, buffer = cv2.imencode(".jpg", output_img)
    base64_img = base64.b64encode(buffer).decode("utf-8")
    return base64_img

@app.post("/vision-search-hf")
async def detect_car(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result_image = predict(image_bytes)
    return JSONResponse(content={"image": result_image})
