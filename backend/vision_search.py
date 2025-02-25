from sklearn.metrics.pairwise import cosine_similarity
import csv
import numpy as np
import ast

from feature_extraction import extract_features, feature_extractor

dataset_features_path = '/app/google_drive/dataset_features.csv'

def load_dataset_features(feature_csv=dataset_features_path):
    """
    Load precomputed features from the CSV file.
    :param feature_csv: Path to the CSV file containing dataset features.
    :return: A dictionary mapping image paths to their feature vectors.
    """
    dataset_features = {}
    with open(feature_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            img_path, features_str = row  # Changed variable name here
            # Convert the string back to a list of floats
            features_list = ast.literal_eval(features_str)
            # Convert the list to a NumPy array
            dataset_features[img_path] = np.array(features_list, dtype=np.float32)
    return dataset_features

def find_similar_images(query_image_path, dataset_features, feature_extractor, top_n=5):
    """
    Find the most similar images to the query image.
    :param query_image_path: Path to the query image.
    :param dataset_features: Precomputed features of the dataset.
    :param feature_extractor: The feature extraction model.
    :param top_n: Number of similar images to return.
    :return: List of tuples (image_path, similarity_score).
    """
    # Extract features from the query image
    query_features = extract_features(query_image_path, feature_extractor)

    # Compute similarity scores
    similarities = []
    for img_path, features in dataset_features.items():
        score = cosine_similarity([query_features], [features])[0][0]
        similarities.append((img_path, score))

    # Sort by similarity score (descending order)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top N similar images
    return similarities[:top_n]

# Example usage
if __name__ == "__main__":
    # Load precomputed dataset features
    dataset_features = load_dataset_features()

    # Path to the query image
    query_image_path = "./subaru.jpg"

    # Find similar images
    similar_images = find_similar_images(query_image_path, dataset_features, feature_extractor, top_n=5)

    # Print the results
    print("Top 5 Similar Images:")
    for img_path, score in similar_images:
        print(f"Image: {img_path}, Similarity Score: {score:.4f}")