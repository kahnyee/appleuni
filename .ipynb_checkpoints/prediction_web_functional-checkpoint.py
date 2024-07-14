import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import matplotlib as mpl
import os

# Constants
IMG_SIZE = (75, 75)
LAST_CONV_LAYER_NAME = "conv2d_24"
CLASSIFIER_LAYER_NAMES = ["avg_pool", "predictions"]
MODEL_PATH = "C:/Users/kahny/ML Model/model_functional.h5"
LABEL_MAP = {0: "It is an apple!", 1: "It is a uni sushi!", 2: "It is unknown!"}

# Helper function to convert image to array
def get_img_array(img, size):
    img = img.convert('RGB')
    array = keras.utils.img_to_array(img)
    array = (array.astype(np.float32) / 255.0)
    array = np.expand_dims(array, axis=0)
    return array

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that outputs the last conv layer and predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Convert preds to a tensor if it is a list
        if isinstance(preds, list):
            preds = tf.convert_to_tensor(preds)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        # Ensure preds is the right shape before indexing
        if len(preds.shape) == 2:
            class_channel = preds[:, pred_index]
        elif len(preds.shape) == 1:
            class_channel = preds[pred_index]
        elif len(preds.shape) == 3 and preds.shape[1] == 1:
            preds = tf.squeeze(preds, axis=1)
            class_channel = preds[:, pred_index]
        else:
            raise ValueError(f"Unexpected shape of preds tensor: {preds.shape}")

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display Grad-CAM heatmap
def save_and_display_gradcam(image, heatmap, caption, cam_path="cam.jpg", alpha=0.4, display_size=(704, 704)):
    img = keras.utils.img_to_array(image)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    st.image(superimposed_img, width=display_size[0], caption=caption)

# Streamlit app
st.write("# Apple, Uni, Unknown Classification")
st.write("This is a simple image classification web app to predict apple, uni, or unknown objects")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    st.image(image, use_column_width=True)
    image = ImageOps.fit(image, IMG_SIZE, Image.LANCZOS)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    img_array = get_img_array(image, size=IMG_SIZE)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    st.write(LABEL_MAP[predicted_label])
    st.text("Probability (0: Apple, 1: Uni, 2: Unknown)")
    st.write(prediction)

    heatmap_apple = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, pred_index=0)
    heatmap_uni = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, pred_index=1)
    heatmap_unknown = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, pred_index=2)

    save_and_display_gradcam(image, heatmap_apple, "Apple Heatmap")
    save_and_display_gradcam(image, heatmap_uni, "Uni Sushi Heatmap")
    save_and_display_gradcam(image, heatmap_unknown, "Unknown Heatmap")