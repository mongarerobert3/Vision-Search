from flask import Flask, request, jsonify
from vision_search import load_dataset_features, find_similar_images, feature_extractor

app = Flask(__name__)

# Load precomputed dataset features
dataset_features = load_dataset_features()

@app.route('/similar-images', methods=['POST'])
def get_similar_images():
    data = request.json
    query_image_path = data.get('query_image_path')
    top_n = data.get('top_n', 5)

    if not query_image_path:
        return jsonify({'error': 'No query image path provided'}), 400

    similar_images = find_similar_images(query_image_path, dataset_features, feature_extractor, top_n)

    return jsonify(similar_images)

if __name__ == "__main__":
    app.run(debug=True)