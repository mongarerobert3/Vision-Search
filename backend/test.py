import roboflow

# Initialize Roboflow
rf = roboflow.Roboflow(api_key="j80P1KwbPUw0s7jWoGkQ")
print("Roboflow initialized successfully")

# Load the project
project = rf.workspace().project("car-brand-detection-eak9j")
print("Project loaded successfully")

model = project.version("1").model


# Load the model
roboflow_model = project.version("1").model
print("Roboflow model loaded successfully")

# Run inference on a local image
prediction = model.predict("./subaru.jpg")

# Convert predictions to JSON
prediction_json = prediction.json()
print("Predictions:", prediction_json)