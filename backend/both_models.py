from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import requests
from pinecone import Pinecone, ServerlessSpec
from decouple import config
from roboflow import Roboflow

app = Flask(__name__)
CORS(app, origins=["https://vision-search-five.vercel.app"])

# Pinecone API key and environment
PINECONE_API_KEY = config("PINECONE_API_KEY") 
PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT") 
PINECONE_INDEX_NAME = "car-images"  

HF_API_URL = "https://api-inference.huggingface.co/models/courte/Car_Vision"
HF_API_TOKEN = config("HF_API_TOKEN")

# Define paths
DATASET_PATH = "./Car_Sales_vision_ai_project"
MODEL_REPO_ID = "courte/Car_Vision" 
MODEL_FILENAME = "car_brand_classifier_final.h5"

# Load the model and feature extractor from Hugging Face
model_name = "courte/Car_Vision"
model_filename = "car_brand_classifier_final_savedmodel/car_brand_classifier_final.h5"

# Load the model directly from Hugging Face
model_path = hf_hub_download(repo_id=model_name, filename=model_filename)
model = tf.keras.models.load_model(model_path)
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=256,  # Ensure this matches your model's embedding size
        metric="cosine",  # Similarity search metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Modify region if needed
    )

# Get index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Roboflow API client
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
roboflow_model = rf.workspace("starter-ccy4i").project("car-brand-detection").version(1).model

def extract_features(image_path):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(HF_API_URL, json={"image": image_data}, headers=headers)

    if response.status_code == 200:
        return np.array(response.json())  
    else:
        raise ValueError(f"HF API Error: {response.text}")

def find_similar_images(query_features, top_n=5):
    # Query Pinecone for similar images
    results = index.query(
        vector=query_features,
        top_k=top_n,
        include_metadata=False  # No need for metadata in this case
    )
    return [(match['id'], match['score']) for match in results['matches']]

def run_roboflow_inference(image_path):
    # Run inference on the image using Roboflow
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust if necessary
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    predictions = roboflow_model.predict(img_array).json()
    return predictions

@app.route('/images/<path:filename>')
def serve_image(filename):
    image_path = os.path.join(DATASET_PATH, filename)
    return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))

@app.route('/vision-search', methods=['POST'])
def vision_search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    query_image = request.files['image']
    if query_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_dir = '/tmp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{secure_filename(query_image.filename)}")
    query_image.save(temp_image_path)

    try:
        # Run inference using both models
        hf_features = extract_features(temp_image_path, feature_extractor).tolist()
        roboflow_predictions = run_roboflow_inference(temp_image_path)

        # Compare predictions (choose the one with the highest confidence or other logic)
        hf_prediction_score = max(hf_features)  # This depends on how your HF model returns predictions
        roboflow_prediction_score = max([pred['confidence'] for pred in roboflow_predictions['predictions']])

        # If Roboflow's prediction is more confident, use it, otherwise fallback to Hugging Face model
        if roboflow_prediction_score > hf_prediction_score:
            result = {
                'model': 'Roboflow',
                'predictions': roboflow_predictions['predictions']
            }
        else:
            # Find similar images from Pinecone using Hugging Face features
            similar_images = find_similar_images(hf_features, top_n=5)
            result = {
                'model': 'HuggingFace',
                'similarImages': [
                    {
                        'url': f"/images/{os.path.relpath(img_path, DATASET_PATH)}",
                        'name': os.path.basename(img_path)
                    }
                    for img_path, score in similar_images
                ]
            }

        return jsonify(result)

    except Exception as e:
        print(f"Error during vision search: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
