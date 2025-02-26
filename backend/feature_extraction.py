import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

# Load the trained model (without the classification head)
model = tf.keras.models.load_model('./car_brand_classifier_final.keras')

# Remove the classification head to get the feature extraction part
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)  # Output before the Dense layers

def extract_features(image_path, feature_extractor):
    """
    Extract features from an image using the feature extractor.
    :param image_path: Path to the image file.
    :param feature_extractor: The feature extraction model.
    :return: A feature vector for the image.
    """
    img = image.load_img(image_path, target_size=(299, 299))  # Resize to match the model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image

    # Extract features
    features = feature_extractor.predict(img_array)
    return features.flatten()  # Flatten the feature vector