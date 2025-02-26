import csv
from tqdm import tqdm
from feature_extraction import extract_features, feature_extractor
import os

# Path to the dataset folder
dataset_path = '/content/drive/My Drive/Car_Sales_vision_ai_project'

# Dictionary to store image paths and their corresponding features
dataset_features = {}

# Iterate through the dataset and extract features
for brand in tqdm(os.listdir(dataset_path)):
    brand_folder = os.path.join(dataset_path, brand)
    if os.path.isdir(brand_folder):
        for img_file in os.listdir(brand_folder):
            img_path = os.path.join(brand_folder, img_file)
            try:
                features = extract_features(img_path, feature_extractor)
                dataset_features[img_path] = features
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Save the dataset features to a CSV file for later use
with open('dataset_features.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'features'])
    for img_path, features in dataset_features.items():
        writer.writerow([img_path, features.tolist()])