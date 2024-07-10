import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2

def get_img_array(img):
    size = (75, 75)
    image = ImageOps.fit(img, size, Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_array = np.expand_dims(image, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

def import_and_predict(image_data, model):
    img_array = get_img_array(image_data)
    prediction = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, 'conv2d_4')  # Use the name of your last convolutional layer
    return prediction, heatmap

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = ImageOps.fit(img, (75, 75), Image.LANCZOS)
    img = img.convert('RGB')
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

# Loading a trained model
model_path = "C:/Users/kahny/ML Model/97bestmodel.h5"
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
    prediction, heatmap = import_and_predict(image, model)

    label_map = {0: "It is an apple!", 1: "It is a uni sushi!", 2: "It is unknown!"}
    predicted_label = np.argmax(prediction)
    st.write(label_map[predicted_label])

    st.text("Probability (0: Apple, 1: Uni, 2: Unknown)")
    st.write(prediction)

    # Display heatmap
    cam_path = save_and_display_gradcam(image, heatmap)
    st.image(cam_path, caption='Grad-CAM Heatmap', use_column_width=True)