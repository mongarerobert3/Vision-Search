from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing import image
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from decouple import config

app = Flask(__name__)
CORS(app, origins=["https://vision-search-five.vercel.app/"])

# Pinecone API key and environment
PINECONE_API_KEY =config("PINECONE_API_KEY") 
PINECONE_ENVIRONMENT =config("PINECONE_ENVIRONMENT") 
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
index = pinecone.Index(PINECONE_INDEX_NAME)

def extract_features(image_path):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(HF_API_URL, json={"image": image_data}, headers=headers)

    if response.status_code == 200:
        return np.array(response.json())  # Expecting an array from the API
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
        query_features = extract_features(temp_image_path, feature_extractor).tolist()

        # Find similar images using Pinecone
        similar_images = find_similar_images(query_features, top_n=5)

        result = [
            {
                'url': f"/images/{os.path.relpath(img_path, DATASET_PATH)}",
                'name': os.path.basename(img_path)
            }
            for img_path, score in similar_images
        ]

        return jsonify({'similarImages': result})

    except Exception as e:
        print(f"Error during vision search: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')