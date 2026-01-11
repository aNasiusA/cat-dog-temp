import os
import gdown
import numpy as np
import streamlit as st
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# # Load your saved model
# model = load_model("cat_dog_model.keras")
MODEL_PATH = "model.keras"
MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1Gze4ACA8UWMWDYmr7XZ0KDLLvqeaFz0G"
)

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
# Our page title
st.title("Cat or Dog Classifier 🐱🐶")

# Upload image
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = load_img(file, target_size=(224, 224))  # Resize to VGG16 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    preds = model.predict(img_array)

    if preds.shape[-1] == 1:
        score = float(preds[0][0])
        if score > 0.5:
            prediction = "Dog"
            confidence = score
        else:
            prediction = "Cat"
            confidence = 1 - score

        st.write(f"Prediction: **{prediction}** with confidence {confidence:.2f}")
