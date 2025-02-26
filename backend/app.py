from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import tensorflow as tf
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Define constants
MODEL_REPO_ID = "courte/Car_Vision"  # Replace with your Hugging Face repo ID
MODEL_FILENAME = "car_brand_classifier_final.h5"
FEATURES_CSV = "dataset_features.csv"

# Load the feature extractor model from Hugging Face
def load_model_from_huggingface(repo_id, filename):
    """
    Download and load the model from Hugging Face.
    :param repo_id: Repository ID on Hugging Face.
    :param filename: Model file name.
    :return: Loaded TensorFlow model.
    """
    print("Downloading model from Hugging Face...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return tf.keras.models.load_model(model_path)

# Load precomputed dataset features
def load_dataset_features(features_csv):
    """
    Load precomputed features from a CSV file.
    :param features_csv: Path to the CSV file containing dataset features.
    :return: A dictionary mapping image paths to their feature vectors.
    """
    dataset_features = {}
    try:
        with open(features_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) != 2:  # Ensure the row has exactly two columns
                    print(f"Skipping invalid row: {row}")
                    continue

                img_path, features_str = row
                # Clean the features string (remove brackets and whitespace)
                features_cleaned = features_str.replace('[', '').replace(']', '').replace(' ', '')

                # Convert cleaned features to a NumPy array
                try:
                    features = np.array(features_cleaned.split(','), dtype=np.float32)
                    dataset_features[img_path] = features
                except ValueError as e:
                    print(f"Error processing features for {img_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File '{features_csv}' not found.")
    except Exception as e:
        print(f"Error loading dataset features: {e}")


# Extract features from an image
def extract_features(image_path, feature_extractor):
    """
    Extract features from an image using the feature extractor.
    :param image_path: Path to the image file.
    :param feature_extractor: The feature extraction model.
    :return: A feature vector for the image.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    features = feature_extractor.predict(img_array)
    return features.flatten()

# Find similar images
def find_similar_images(query_features, dataset_features, top_n=5):
    """
    Find the most similar images based on cosine similarity.
    :param query_features: Feature vector of the query image.
    :param dataset_features: Dictionary of precomputed dataset features.
    :param top_n: Number of similar images to return.
    :return: List of tuples (image_path, similarity_score).
    """
    similarities = []
    for img_path, features in dataset_features.items():
        score = cosine_similarity([query_features], [features])[0][0]
        similarities.append((img_path, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Initialize the model and dataset features
model = load_model_from_huggingface(MODEL_REPO_ID, MODEL_FILENAME)
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
dataset_features = load_dataset_features(FEATURES_CSV)

@app.route('/vision-search', methods=['POST'])
def vision_search():
    """
    Endpoint for vision search.
    Receives an image, extracts its features, and returns similar images.
    """
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
        similar_images = find_similar_images(query_features, dataset_features, top_n=5)

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
      app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
