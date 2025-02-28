from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import tempfile
import numpy as np
import requests
import tensorflow as tf
from huggingface_hub import hf_hub_download
from werkzeug.utils import secure_filename
from decouple import config
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://vision-search-five.vercel.app"}})

# Load API keys
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "car-images"
HF_API_URL = "https://api-inference.huggingface.co/models/courte/Car_Vision"
HF_API_TOKEN = config("HF_API_TOKEN")
SPACE_API_URL = "https://courte-car-vision.hf.space/api/predict"

# Hugging Face model details
MODEL_REPO_ID = "courte/Car_Vision"
MODEL_FILENAME = "car_brand_classifier_final_savedmodel/car_brand_classifier_final.h5"

# Download and load model
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = tf.keras.models.load_model(model_path)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=256,  # Match model's embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

def preprocess_image(image_obj):
    """
    Preprocess image: Convert to RGB, resize, normalize, and add batch dimension.
    """
    if isinstance(image_obj, str):  # If filepath is provided
        image_obj = Image.open(image_obj)

    image_obj = image_obj.convert("RGB")  # Ensure RGB format
    image_obj = image_obj.resize((299, 299))  # Match model input size
    image_arr = np.array(image_obj) / 255.0  # Normalize
    return np.expand_dims(image_arr, axis=0)

def image_to_base64(image_obj):
    """
    Convert a PIL Image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve images from the dataset directory.
    """
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

@app.route('/vision-search', methods=['POST'])
def vision_search():
    """
    Process an uploaded image, preprocess it locally, and send it to the Spaces API for feature extraction.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    uploaded_image = request.files['image']
    if uploaded_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the uploaded image
        img = Image.open(uploaded_image).convert("RGB")

        # Preprocess the image locally
        preprocessed_image = preprocess_image(img)

        # Convert the preprocessed image to a base64-encoded string
        preprocessed_image_pil = Image.fromarray((preprocessed_image[0] * 255).astype(np.uint8))
        image_base64 = image_to_base64(preprocessed_image_pil)

        # Send the preprocessed image to the Spaces API
        response = requests.post(SPACE_API_URL, json={"image": image_base64})

        if response.status_code != 200:
            raise ValueError(f"Space API Error: {response.text}")

        # Extract features from the API response
        features = np.array(response.json())  # Ensure this matches the expected format

        # Query Pinecone for similar images
        results = find_similar_images(features, top_n=5)

        # Prepare the response
        similar_images = [
            {
                'id': match['id'],
                'score': float(match['score']),
                'url': f"/images/{match['id']}"  # Serve images from the dataset directory
            }
            for match in results
        ]
        return jsonify({'similarImages': similar_images})

    except Exception as e:
        print(f"Error during vision search: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')