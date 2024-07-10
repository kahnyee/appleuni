from PIL import Image, ImageOps
import tensorflow as tf
import cv2
import numpy as np
import os
import sys

label = ''
frame = None

def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load the pre-trained model
model_path = 'C:/Users/kahny/ML Model/97bestmodel.h5'
model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera OK")
else:
    cap.open()

while True:
    ret, original = cap.read()
    if not ret:
        break

    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    # Make prediction
    prediction = import_and_predict(image, model)

    # Determine the prediction result
    if np.argmax(prediction) == 0:
        predict = "It is an apple!"
    elif np.argmax(prediction) == 1:
        predict = "It is a uni sushi!"
    else:
        predict = "It is unknown!"

    # Display the prediction on the video frame
    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
sys.exit()