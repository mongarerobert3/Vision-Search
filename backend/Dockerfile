# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install required libraries including gdown
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt gdown

# Copy the rest of the application code into the container
COPY . .

# Download model and dataset from Google Drive
RUN gdown https://drive.google.com/uc?id=1Sq5pX7-KIgs-C-NbdVVW_soNI-uG_IlL
RUN gdown https://drive.google.com/uc?id=1Oh5kzEb-Mv9_-0Q2ulANPpHlQp4FcHd7

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
