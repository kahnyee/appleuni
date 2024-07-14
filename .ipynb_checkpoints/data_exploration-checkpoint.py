import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Constants for image height and width
IMG_HEIGHT = 300  
IMG_WIDTH = 300   

def read_and_decode(filename, reshape_dims):
    # Reads an image file, decodes it, and resizes it to the specified dimensions
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, reshape_dims)
    return img

def show_image(filename, reshape_dims):
    # Displays an image with given filename after reading and resizing it
    img = read_and_decode(filename, reshape_dims)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(os.path.basename(filename))
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.grid(False)

def plot_images_from_folders(base_folder, folders, reshape_dims):
    # Plots images from multiple folders within a base directory
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    show_image(file_path, reshape_dims)
                    plt.show()

base_folder = 'C:/Users/kahny/ML Model'
folders = ['Apple_Resized', 'Uni Sushi_Resized', 'Unknown_Resized']

# Plot images from the specified folders
plot_images_from_folders(base_folder, folders, [IMG_HEIGHT, IMG_WIDTH])