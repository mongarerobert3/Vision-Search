from transformers import pipeline

classifier = pipeline("image-classification", model="courte/Car_Vision")

result = classifier("./subaru.jpg")  # Replace with actual image
print(result)
