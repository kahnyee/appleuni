import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    """Processes an image and makes a prediction using the given model."""
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Loading a trained model
model_path = "C:/Users/kahny/ML Model/model_sequential.h5"
model = tf.keras.models.load_model(model_path)

# Streamlit app
st.write("# Apple, Uni, Unknown Classification")
st.write("This is a simple image classification web app to predict apple, uni, or unknown objects")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    # Mapping the prediction to the corresponding label
    label_map = {0: "It is an apple!", 1: "It is a uni sushi!", 2: "It is unknown!"}
    predicted_label = np.argmax(prediction)
    st.write(label_map[predicted_label])

    st.text("Probability (0: Apple, 1: Uni, 2: Unknown)")
    st.write(prediction)