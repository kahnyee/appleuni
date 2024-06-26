from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def image_gen_w_aug(train_parent_directory, test_parent_directory, validate_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      rotation_range = 30,  
                                      zoom_range = 0.2, 
                                      width_shift_range=0.2,  
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    val_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size = (75,75),
                                                        batch_size = 50,
                                                        class_mode = 'categorical',)
    
    val_generator = val_datagen.flow_from_directory(validate_parent_directory,
                                                          target_size = (75,75),
                                                          batch_size = 15,
                                                          class_mode = 'categorical',)
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size=(75,75),
                                                      batch_size = 15,
                                                      class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    # x = Conv2D(32, (3,3), activation='relu', padding='same')(last_output)
    # x = MaxPooling2D((2,2))(x)
    # x = BatchNormalization()(x)    

    # x = Conv2D(64, (3,3), activation='relu', padding='same')(last_output)
    # x = MaxPooling2D((2,2))(x)
    # x = BatchNormalization()(x)
    
    x = Flatten()(last_output)
    
    
    # Dense hidden layer with Batch Normalization and L2 regularization
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # # Additional Dense layer
    # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    
    # Output neuron. 
    x = Dense(3, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model

# Hyperparameters
lr = 0.0001
epochs = 40
steps_per_epoch = 50
l2 = 0.00001

root = 'C:/Users/xcomb/OneDrive/Desktop/ML project/appleuni'

train_dir = os.path.join(root+'/Train_Resized/')
val_dir = os.path.join(root+'/Validate_Resized/')
test_dir = os.path.join(root+'/Test_Resized/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir, val_dir)

pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history_TL = model_TL.fit(
      train_generator,
      steps_per_epoch=steps_per_epoch,  
      epochs=epochs,
      verbose=1,
      validation_data = validation_generator,
      callbacks=[]
      )

# Evaluate the model on test data
test_loss, test_acc = model_TL.evaluate(test_generator, verbose=1)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_TL.history['accuracy'], label='Training Accuracy')
plt.plot(history_TL.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_TL.history['loss'], label='Training Loss')
plt.plot(history_TL.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

tf.keras.models.save_model(model_TL,'my_model.hdf5')