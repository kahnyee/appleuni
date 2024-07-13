import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import regularizers

# Hyperparameters
epochs = 100
batch_train_size = 16
batch_size = 15
initial_learning_rate = 1.5e-4
l2_regularize = 1.5e-5
l1_regularize = 1e-6

# Directories
root = 'C:/Users/kahny/ML Model'
train_dir = os.path.join(root, 'Train_Resized/')
val_dir = os.path.join(root, 'Validate_Resized/')
test_dir = os.path.join(root, 'Test_Resized/')

# Function to create data generators
def image_gen_w_aug(train_parent_directory, validate_parent_directory, test_parent_directory):
    # Data augmentation for the training data
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=35,
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and test data
    val_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_parent_directory,
        target_size=(75, 75),
        batch_size=batch_train_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        validate_parent_directory,
        target_size=(75, 75),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_parent_directory,
        target_size=(75, 75),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# Create data generators
train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

# Function to build the model which attempts to simulate the InceptionV3 layers without using Transfer Learning.
def build_sequential_inception(input_shape):
    model = Sequential()

    # Initial Conv Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # First Inception-like block
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu'))
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Second Inception-like block
    model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
    model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Third Inception-like block
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(192, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Fourth Inception-like block
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(192, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(BatchNormalization())
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regularize, l2=l2_regularize)))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regularize, l2=l2_regularize)))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes for classification

    return model

# Initialize model
input_shape = (75, 75, 3)
model = build_sequential_inception(input_shape)

# Custom Callback to stop training when both training and validation accuracy reach 97%
class CustomStopper(Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc is not None and val_acc is not None:
            if train_acc >= 0.97 and val_acc >= 0.97:
                print("Stopping training as both training and validation accuracy have reached 97%.")
                self.model.stop_training = True

# Compile the model
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

# Print model summary
model.summary()

# Train the model
with tf.device("/GPU:0"):
    history = model.fit(
        train_dataset,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_dataset,
        validation_steps=len(validation_generator),
        callbacks=[reduce_lr, custom_stopper]
    )

# Save the model
model.save('my_model.h5')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

# Print summary
print(f"\nModel Summary:")
print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Test Loss: {test_loss:.4f}\n")
print(f"Learning Rate: {initial_learning_rate}")
print(f"Epochs: {epochs}")
print(f"Steps Per Epoch: {len(train_generator)}")
print(f"Batch Size: {train_generator.batch_size}")

# Plot accuracy and loss graphs
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