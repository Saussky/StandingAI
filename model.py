import tensorflow as tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image # TODO: same import? might be a bit redundant
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from image_processing import load_and_preprocess_image, classify_images


image_path = './images'
batch_size = 5

# Initialize an image data generator with some data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    rotation_range=40,    # Random rotations
    width_shift_range=0.2, # Random horizontal shifting
    height_shift_range=0.2, # Random vertical shifting
    shear_range=0.2,      # Shear transformations
    zoom_range=0.2,       # Zooming
    horizontal_flip=True, # Horizontal flipping
    brightness_range=[0.8, 1.2],  # Adjust brightness
    channel_shift_range=20.0,     # Shift channel values
    fill_mode='nearest',  # Strategy for filling newly created pixels
    validation_split=0.2  # Split data into training (80%) and validation (20%)
)

# Prepare training and validation data
train_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=(224, 224),  # Resize images to 200x200
    batch_size=5,
    class_mode='binary',    # Binary labels (sitting/standing)
    subset='training'       # Specify this is training data
)

validation_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=(224, 224),
    batch_size=5,
    class_mode='binary',
    subset='validation'     # Specify this is validation data
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze all the layers in the base model
for layer in base_model.layers:
    layer.trainable = False


model = models.Sequential([
    base_model,  # Add the base model as the first layer
    layers.GlobalAveragePooling2D(),  # Add global average pooling layer
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
)


# classify_images(model, './images/standing')
# classify_images(model, './predict')