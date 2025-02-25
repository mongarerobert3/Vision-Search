from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
from flask_caching import Cache

from vision_search import load_dataset_features, find_similar_images
from feature_extraction import extract_features, feature_extractor
import tensorflow as tf

app = Flask(__name__)

# Define the paths to the model and dataset features
model_path = '/app/google_drive/car_brand_classifier_final.keras'
dataset_features_path = '/app/google_drive/dataset_features.csv'

# Load the feature extractor model
model = tf.keras.models.load_model(model_path)
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# Load precomputed dataset features
dataset_features = load_dataset_features(model_path)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/vision-search', methods=['POST'])
@cache.cached(timeout=60, key_prefix='vision_search_')
def vision_search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    query_image = request.files['image']
    if query_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image temporarily
    temp_dir = '/tmp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{secure_filename(query_image.filename)}")
    query_image.save(temp_image_path)

    try:
        # Extract features from the query image
        query_features = extract_features(temp_image_path, feature_extractor)

        # Find similar images
        similar_images = find_similar_images(temp_image_path, dataset_features, feature_extractor, top_n=5)

        # Prepare the response
        result = [
            {'url': img_path, 'similarity': score}
            for img_path, score in similar_images
        ]

        return jsonify({'similarImages': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(debug=True)