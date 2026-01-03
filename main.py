import os
import json
import requests
from PIL import Image
from tensorflow.keras import layers,models

import numpy as np
import tensorflow as tf
import streamlit as st
img_size=224

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_models/plant_disease_model.weights.h5"
# Load the pre-trained model
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(38,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.load_weights(model_path)
# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


def inference(prompt):
    # USE THE EXACT NAME FROM YOUR SCREENSHOT
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    }

    try:
        # Use host.docker.internal to reach Windows from Docker
        r = requests.post("http://host.docker.internal:11434/api/generate", json=payload, timeout=120)

        # This will verify if we got a 200 OK from the server
        r.raise_for_status()

        # Get the JSON
        data = r.json()

        # If 'response' is missing, return the whole dictionary so we can see the error
        if 'response' not in data:
            return {"response": f"DEBUG ERROR: Ollama returned: {data}"}

        return data

    except requests.exceptions.ConnectionError:
        return {
            "response": "CONNECTION ERROR: Docker cannot reach Ollama. 1. Check if Ollama is running. 2. Set OLLAMA_HOST=0.0.0.0"}
    except Exception as e:
        return {"response": f"PYTHON ERROR: {str(e)}"}

    response = r.json()
    print(response)
    return response

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(mmodel, image_path, cllass_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = mmodel.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = cllass_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            prompt = f'''plant Detected Disease(based on images of leaves): {prediction}

### INSTRUCTIONS ###
Based on the data above, please provide:
1. A brief description of what this disease/status means for the plant.
2. If UNHEALTHY: Immediate steps to stop the spread and a treatment plan.
3. If HEALTHY: A 3-point maintenance checklist to keep it thriving.
4. Risk Assessment: How likely is this to spread to nearby plants (High/Medium/Low)?
5.do not write any other texts other than the given instructions
6.plant names can be any name of vegtebles or fruits like potato,blueberry etc'''
            st.success(f'Prediction: {str(prediction)}')
            with st.spinner('Analysing...'):
                result = inference(prompt) # result is a dictionary

                # Extract the actual text message
                final_text = result.get('response', 'No response found')

                # Display nicely in the app
                st.write(final_text)

                # Save to file safely
                with open("response.txt", "w") as f:
                    f.write(str(final_text))