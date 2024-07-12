import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Constants
IMG_HEIGHT = 300  # You can set this to your desired height
IMG_WIDTH = 300   # You can set this to your desired width

def read_and_decode(filename, reshape_dims):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, reshape_dims)
    return img

def show_image(filename, reshape_dims):
    img = read_and_decode(filename, reshape_dims)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(filename)
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.grid(False)

def plot_images_from_folders(base_folder, folders, reshape_dims):
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    show_image(file_path, reshape_dims)
                    plt.show()

# Define your base folder and folders containing images
base_folder = 'C:/Users/Jayden/Desktop/ml_project_github/appleuni'
folders = ['Apple_Resized', 'Uni Sushi_Resized', 'Unknown_Resized']

# Plot images
plot_images_from_folders(base_folder, folders, [IMG_HEIGHT, IMG_WIDTH])