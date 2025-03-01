from gradio_client import Client
from PIL import Image
import base64

def image_to_base64(image_path):
    """
    Convert an image file to a base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to the image file
image_path = "subaru1.jpg"

# Convert the image to a base64-encoded string
image_base64 = image_to_base64(image_path)

# Create a Gradio Client to call the Hugging Face Space
client = Client("courte/Car_Vision")

# Make the API call using the base64 image string
result = client.predict(
    {
        "data": [
            f"data:image/jpeg;base64,{image_base64}"
        ]
    },
    api_name="/predict" 
)

print(result)