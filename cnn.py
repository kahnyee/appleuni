from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

# Hyperparameters
epochs = 100
batch_train_size = 16
batch_size = 15
initial_learning_rate = 1.5e-4
l2_regularize = 1.5e-5
l1_regularize = 1e-6

con_layer = 4
con_filters = "(64,128,256,512)"

dense_layer = 3
drop_out_rate = "(0.1,0.1,0.1)"
dense_filters="(64,32,16)"

regularizer = 4
data_aug = True

# Directories
root = 'C:/Users/kahny/ML Model'
train_dir = os.path.join(root, 'Train_Resized/')
val_dir = os.path.join(root, 'Validate_Resized/')
test_dir = os.path.join(root, 'Test_Resized/')

def image_gen_w_aug(train_parent_directory, validate_parent_directory, test_parent_directory):
    train_datagen = ImageDataGenerator(rescale=1/255,
                                       rotation_range=45,
                                       zoom_range=0.3,
                                       width_shift_range=0.3,
                                       height_shift_range=0.3,
                                       shear_range=0.3,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1/255)
    val_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size=(75, 75),
                                                        batch_size=batch_train_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    val_generator = val_datagen.flow_from_directory(validate_parent_directory,
                                                    target_size=(75, 75),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size=(75, 75),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    return train_generator, val_generator, test_generator

def inception_module(x, filters):
    f1, f2, f3, f4 = filters
    
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)
    
    conv3 = Conv2D(f2[0], (1, 1), padding='same', activation='relu')(x)
    conv3 = Conv2D(f2[1], (3, 3), padding='same', activation='relu')(conv3)
    
    conv5 = Conv2D(f3[0], (1, 1), padding='same', activation='relu')(x)
    conv5 = Conv2D(f3[1], (5, 5), padding='same', activation='relu')(conv5)
    
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = Conv2D(f4, (1, 1), padding='same', activation='relu')(pool)
    
    out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return out

def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    x = inception_module(x, [32, (48, 64), (8, 16), 16])
    x = MaxPooling2D((2, 2))(x)
    
    x = inception_module(x, [64, (96, 128), (16, 32), 32])
    x = MaxPooling2D((2, 2))(x)
    
    x = inception_module(x, [128, (128, 192), (32, 96), 64])
    x = MaxPooling2D((2, 2))(x)
    
    x = inception_module(x, [128, (128, 192), (32, 96), 64])
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regularize, l2=l2_regularize))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regularize, l2=l2_regularize))(x)
    x = Dropout(0.2)(x)
    
    output_layer = Dense(3, activation='softmax')(x)
    
    model = Model(input_layer, output_layer)
    return model

class CustomStopper(Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc is not None and val_acc is not None:
            if train_acc >= 0.97 and val_acc >= 0.97:
                print(f"\nStopping training as both train and val accuracy have reached 97%.")
                self.model.stop_training = True

def plot_misclassified_images(generator, true_labels, predicted_classes, class_labels, img_dir, title):
    # Find the indices of misclassified images
    misclassified_indices = np.where(predicted_classes != true_labels)[0]
    
    # Number of misclassified images
    num_misclassified = len(misclassified_indices)
    print(f"Number of misclassified images in {title}: {num_misclassified}")
    
    if num_misclassified > 0:
        max_images_per_plot = 9
        num_plots = int(np.ceil(num_misclassified / max_images_per_plot))
        
        for plot_num in range(num_plots):
            start_idx = plot_num * max_images_per_plot
            end_idx = start_idx + max_images_per_plot
            plot_indices = misclassified_indices[start_idx:end_idx]
            
            num_images_in_plot = len(plot_indices)
            rows = int(np.ceil(num_images_in_plot / 3))
            cols = 3  # Fixed number of columns
            
            plt.figure(figsize=(15, 15))
            for i, idx in enumerate(plot_indices):
                plt.subplot(rows, cols, i + 1)
                img_path = os.path.join(img_dir, generator.filenames[idx])
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.title(f"True: {class_labels[true_labels[idx]]}\nPred: {class_labels[predicted_classes[idx]]}")
                plt.axis('off')
            plt.suptitle(f"{title} - Plot {plot_num + 1}")
            plt.tight_layout()
            plt.show()
    else:
        print(f"No misclassified images to display in {title}.")

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

input_shape = (75, 75, 3)
model = build_model(input_shape)
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
custom_stopper = CustomStopper()

# Function to convert DirectoryIterator to tf.data.Dataset
def generator_to_dataset(generator):
    output_signature = (
        tf.TensorSpec(shape=(None, 75, 75, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
    )
    def generator_function():
        for batch in generator:
            yield batch
    return tf.data.Dataset.from_generator(generator_function, output_signature=output_signature)

# Convert the generators to tf.data.Dataset
train_dataset = generator_to_dataset(train_generator).repeat()
validation_dataset = generator_to_dataset(validation_generator).repeat()

# Training the model
history = model.fit(
      train_dataset,
      steps_per_epoch=len(train_generator),
      epochs=epochs,
      validation_data=validation_dataset,
      validation_steps=len(validation_generator),
      callbacks=[reduce_lr, custom_stopper]
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]
training_loss = history.history['loss'][-1]
validation_loss = history.history['val_loss'][-1]

# Print summary
print(f"\nModel Summary:")
print(f"Training Accuracy: {training_accuracy:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")
print(f"Training Loss: {training_loss:.4f}")
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}\n")
print(f"Learning Rate: {initial_learning_rate}")
print(f"Epochs: {epochs}")
print(f"Steps Per Epoch: {len(train_generator)}")
print(f"Batch Size: {train_generator.batch_size}")

# Plot the accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save('my_model.h5')

# Plot misclassified images for training, validation, and test sets
def evaluate_and_plot_misclassified(generator, img_dir, title):
    generator.reset()
    true_labels = generator.classes
    class_labels = list(generator.class_indices.keys())
    predictions = model.predict(generator, steps=len(generator), verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    plot_misclassified_images(generator, true_labels, predicted_classes, class_labels, img_dir, title)

evaluate_and_plot_misclassified(train_generator, train_dir, "Training Set")
evaluate_and_plot_misclassified(validation_generator, val_dir, "Validation Set")
evaluate_and_plot_misclassified(test_generator, test_dir, "Test Set")