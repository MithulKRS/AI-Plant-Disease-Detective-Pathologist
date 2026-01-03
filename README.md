# AI-Plant-Disease-Detective-Pathologist
An end-to-end Computer Vision + Generative AI application that detects plant diseases from leaf images and provides immediate, actionable treatment advice using a local Large Language Model (LLM).

🚀 Features
Accurate Detection: Custom CNN (Convolutional Neural Network) trained on plant leaf datasets to classify healthy vs. diseased leaves.

AI Pathologist: Integrated Ollama (Llama 3.2) to generate "Botanist-Level" advice. It explains the disease, suggests cures, and provides prevention tips.

Robust Preprocessing: Uses ImageDataGenerator for real-time data augmentation (rotation, flipping, zoom) to handle real-world lighting and angles.

Containerized Deployment: Fully Dockerized application. Runs consistently on any machine without dependency issues.

User-Friendly UI: Clean interface built with Streamlit for easy image uploading and report generation.

🛠️ Tech Stack
Deep Learning: TensorFlow, Keras (CNN architecture).

GenAI / LLM: Ollama (running Llama 3.2 or TinyLlama locally).

Backend/Frontend: Streamlit, Python.

DevOps: Docker (Dockerfile & networking).

Image Processing: Pillow (PIL), NumPy.

🏗️ System Architecture
This project uses a hybrid architecture where the Vision Model runs inside a Docker container, communicating with a Local LLM running on the host machine.

Input: User uploads an image via Streamlit.

Vision Analysis: The CNN model processes the image and predicts the class (e.g., Apple___Black_rot).

Prompt Engineering: The prediction is wrapped in a structured prompt (Context Injection).

LLM Inference: The Docker container sends this prompt to the host's Ollama instance via host.docker.internal.

Output: The LLM returns a structured treatment plan displayed to the user.

💻 Installation & Setup
Prerequisites
Docker Desktop installed and running.

Ollama installed on your local machine.

Step 1: Configure Ollama (Crucial Step)
By default, Ollama only listens to localhost. To let Docker talk to it, you must enable external connections.

On Windows (PowerShell):

PowerShell

# 1. Quit Ollama from the taskbar tray first!
# 2. Set the environment variable
$env:OLLAMA_HOST = "0.0.0.0"
# 3. Start Ollama
ollama serve
Pull the model:

PowerShell

ollama pull llama3.2:latest
# OR for faster performance on lower-end hardware:
ollama pull tinyllama
Step 2: Clone & Build
Bash

git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Build the Docker image
docker build -t plant-disease-detection-image:v1.0 .
Step 3: Run the Application
Bash

# Map port 80 inside container to port 80 on your machine
docker run -p 80:80 plant-disease-detection-image:v1.0
Step 4: Access the App
Open your browser and go to: http://localhost


