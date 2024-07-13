import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
IMG_SIZE = (75, 75)
MODEL_PATH = "C:/Users/kahny/ML Model/97bestmodelspyder.h5"
ALPHA = 0.6  # Transparency for heatmap

# Function to load and preprocess the image
def preprocess_image(image, size):
    image = image.resize(size, Image.LANCZOS).convert('RGB')
    array = np.array(image) / 255.0  # Convert PIL Image to numpy array and normalize
    return np.expand_dims(array, axis=0)

# Function to find last convolutional layer
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional layer found in the model.")

# Function to generate Grad-CAM heatmap
def get_grad_cam(model, img_array, class_idx):
    last_conv_layer = model.get_layer('conv2d_130')  # Adjusted to your last conv layer
    conv_output = last_conv_layer.output

    grad_model = tf.keras.models.Model([model.inputs], [conv_output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8  # Normalize to [0, 1]
    return heatmap.numpy()

# Function to save and display Grad-CAM heatmap
def save_and_display_gradcam(image, heatmap, caption, alpha=0.6, display_size=(704, 704)):
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Apply a colormap and remove the alpha channel
    heatmap_colored = np.uint8(heatmap_colored * 255)  # Convert to uint8
    heatmap_colored = Image.fromarray(heatmap_colored).resize((image.shape[1], image.shape[0]))
    heatmap_colored = np.array(heatmap_colored)
    superimposed_img = heatmap_colored * alpha + image
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))
    st.image(superimposed_img, caption=caption, width=display_size[0])

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

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
    image_resized = image.resize(IMG_SIZE, Image.LANCZOS)
    img_array = preprocess_image(image_resized, IMG_SIZE)

    # Make prediction
    try:
        prediction = model.predict(img_array)
        class_names = {0: "Apple", 1: "Uni Sushi", 2: "Unknown"}
        predicted_class = np.argmax(prediction)
        st.write(f"Prediction: {class_names[predicted_class]}")
        st.text("Probability")
        st.write(prediction)

        # Generate Grad-CAM heatmap
        heatmap = get_grad_cam(model, img_array, predicted_class)

        # Display Grad-CAM heatmap
        save_and_display_gradcam(np.array(image), heatmap, f"{class_names[predicted_class]} Grad-CAM")

    except Exception as e:
        st.text(f"Error: {e}")
