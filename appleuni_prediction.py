import numpy as np
import streamlit as st
import tensorflow as tf
from keras import utils as keras_utils
from PIL import Image, ImageOps
import matplotlib as mpl

# Image size for the model
img_size = (75, 75)

# Names of the last convolutional layer and classifier layers
last_conv_layer_name = "conv2d_174"

def get_img_array(img, size):
    """Convert image to a preprocessed array."""
    img = img.convert('RGB')
    array = keras_utils.img_to_array(img)
    array = array.astype(np.float32) / 255.0
    return np.expand_dims(array, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(image, heatmap, caption, cam_path="cam.jpg", alpha=0.4):
    """Save and display the Grad-CAM heatmap."""
    img = keras_utils.img_to_array(image)
    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras_utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras_utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras_utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    st.image(superimposed_img, use_column_width=True, caption=caption)

# Streamlit web app
st.write("# Apple, Uni, Unknown Classification")
st.write("This is a simple image classification web app to predict apple, uni, or unknown objects")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file:
    image = Image.open(file)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    st.image(image, use_column_width=True)
    image = ImageOps.fit(image, img_size, Image.LANCZOS)
    
    model_path = "C:/Users/Jayden/Desktop/ml_project_github/appleuni/97bestmodel.h5"
    model = tf.keras.models.load_model(model_path)

    img_array = get_img_array(image, size=img_size)
    prediction = model.predict(img_array)
    
    label_map = {0: "It is an apple!", 1: "It is a uni sushi!", 2: "It is unknown!"}
    predicted_label = np.argmax(prediction)
    st.write(label_map[predicted_label])
    st.write("Probability (0: Apple, 1: Uni, 2: Unknown)", prediction)
    
    heatmap_apple = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0)
    heatmap_uni = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=1)
    heatmap_unknown = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=2)

    save_and_display_gradcam(image, heatmap_apple, "Apple Heatmap")
    save_and_display_gradcam(image, heatmap_uni, "Uni Sushi Heatmap")
    save_and_display_gradcam(image, heatmap_unknown, "Unknown Heatmap")
else:
    st.text("You haven't uploaded an image file")