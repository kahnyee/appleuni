from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import tensorflow as tf
import os

print(device_lib.list_local_devices())

# def image_gen_w_aug(train_parent_directory, test_parent_directory, validate_parent_directory):
    
#     train_datagen = ImageDataGenerator(rescale=1/255,
#                                       rotation_range = 30,  
#                                       zoom_range = 0.2, 
#                                       width_shift_range=0.1,  
#                                       height_shift_range=0.1,
#                                       shear_range=0.2)
    
#     test_datagen = ImageDataGenerator(rescale=1/255)
    
#     val_datagen = ImageDataGenerator(rescale=1/255)
    
#     train_generator = train_datagen.flow_from_directory(train_parent_directory,
#                                                         target_size = (75,75),
#                                                         batch_size = 15,
#                                                         class_mode = 'categorical',)
    
#     val_generator = val_datagen.flow_from_directory(validate_parent_directory,
#                                                           target_size = (75,75),
#                                                           batch_size = 15,
#                                                           class_mode = 'categorical',)
    
#     test_generator = test_datagen.flow_from_directory(test_parent_directory,
#                                                       target_size=(75,75),
#                                                       batch_size = 15,
#                                                       class_mode = 'categorical')
    
#     return train_generator, val_generator, test_generator

# root = 'C:/Users/Jayden/Desktop/ml_project_github/appleuni'

# train_dir = os.path.join(root+'/Train/')
# val_dir = os.path.join(root+'/Validate/')
# test_dir = os.path.join(root+'/Test/')

# train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir, val_dir)

# img_size=(75,75)

# model = Sequential()

# model.add(Conv2D(32, (3,3), activation='relu',input_shape=(img_size[0],img_size[1],3)))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(3,activation='softmax'))


# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# with tf.device("/GPU:0"):
#     history = model.fit(
#           train_generator,
#           steps_per_epoch=1,  
#           epochs=50,
#           verbose=1,
#           validation_data = validation_generator)

# tf.keras.models.save_model(model,'my_model.hdf5')



def image_gen_w_aug(train_parent_directory, test_parent_directory, validate_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      rotation_range = 30,  
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      shear_range=0.2)
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    val_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size = (75,75),
                                                        batch_size = 25,
                                                        class_mode = 'categorical',)
    
    val_generator = val_datagen.flow_from_directory(validate_parent_directory,
                                                          target_size = (75,75),
                                                          batch_size = 25,
                                                          class_mode = 'categorical',)
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size=(75,75),
                                                      batch_size = 25,
                                                      class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output neuron. 
    x = Dense(3, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model

root = 'C:/Users/Jayden/Desktop/ml_project_github/appleuni'

train_dir = os.path.join(root+'/Train/')
val_dir = os.path.join(root+'/Validate/')
test_dir = os.path.join(root+'/Test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir, val_dir)

pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_TL = model_TL.fit(
      train_generator,
      steps_per_epoch=10,  
      epochs=25,
      verbose=1,
      validation_data = validation_generator)

tf.keras.models.save_model(model_TL,'my_model.hdf5')





