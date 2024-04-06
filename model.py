# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from utils import crop_and_preprocess_image, crop_image, preprocess_image

# base_dir = './images'

# # Prepare the data augmentation generator
# train_datagen = ImageDataGenerator(
#     preprocessing_function=lambda x: crop_and_preprocess_image(x, crop_image, preprocess_image, target_size=(224, 224)),
#     shear_range=0.2,
#     zoom_range=0.2,
#     brightness_range=[0.8, 1.2],
#     channel_shift_range=20.0,
#     fill_mode='nearest',
#     validation_split=0.2  # Split data into 80% training and 20% validation
# )

# # Prepare training and validation data
# train_generator = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=(224, 224),
#     batch_size=5,
#     class_mode='binary',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=(224, 224),
#     batch_size=5,
#     class_mode='binary',
#     subset='validation'
# )

# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# for layer in base_model.layers:
#     layer.trainable = False

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),  # Add global average pooling layer
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=20,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
# )

# # Plot training and validation accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('accuracy_graph.png')

# # Plot training and validation loss
# plt.figure()  # Create a new figure for the loss graph
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('loss_graph.png')
