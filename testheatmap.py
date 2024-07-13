import numpy as np
import streamlit as st
import tensorflow as tf
from keras import utils as keras_utils
from PIL import Image, ImageOps
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import keras
from tensorflow.keras.layers import Input


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = array.astype(np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    preds = new_model.predict(img_array)
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# last_conv_layer_name = "conv2d_77"
last_conv_layer_name = "conv2d_174"
img_size = (75, 75)


old_model_path = "C:/Users/Jayden/Desktop/ml_project_github/appleuni/97bestmodel.h5"
old_model = tf.keras.models.load_model(old_model_path)


model_path = "C:/Users/Jayden/Desktop/ml_project_github/appleuni/bestmodel.h5"
model = tf.keras.models.load_model(model_path)


input_shape = (75, 75, 3)  # Adjust this to match your required input shape
new_input = Input(input_shape)

# Pass the new input through each layer of the Sequential model individually
x = new_input
for layer in model.layers:
    x = layer(x)

new_model = tf.keras.models.Model(inputs=new_input, outputs=x)
new_model.summary()

image_path = "C:/Users/Jayden/Downloads/uni.jpg"

img_array = get_img_array(image_path, size=img_size)

preds = new_model.predict(img_array)
label_map = {0: "It is an apple!", 1: "It is a uni sushi!", 2: "It is unknown!"}
predicted_label = np.argmax(preds)
print(label_map[predicted_label])


heatmap = make_gradcam_heatmap(img_array, old_model, last_conv_layer_name)






