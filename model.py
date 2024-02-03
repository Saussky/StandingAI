import tensorflow as tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image # TODO: same import? might be a bit redundant
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

image_path = './images'


def crop_and_preprocess(image_array, target_size=(224, 224)):
    """
    Crop the image to focus on the specific area of interest, resize it to the target size,
    and apply MobileNetV2 preprocessing.

    Args:
    - image_array: The image array (NumPy array).
    - target_size: The target size to resize the image after cropping.

    Returns:
    - A NumPy array of the preprocessed image.
    """
    # Crop the image
    height, width = image_array.shape[:2]
    start_row = 0
    start_col = width // 8
    end_row = height // 3
    end_col = 7 * width // 8  # Adjusted to ensure the cropped area maintains aspect ratio
    cropped_image = image_array[start_row:end_row, start_col:end_col, :]
    
    # Resize the cropped image to the target size
    resized_image = tf.image.resize(cropped_image, target_size)
    
    # Apply MobileNetV2 preprocessing
    preprocessed_image = preprocess_input(resized_image)
    
    return preprocessed_image.numpy()  # Convert back to NumPy array if needed



# Genrate training data with data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: crop_and_preprocess(x, target_size=(224, 224)),
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    fill_mode='nearest',
    validation_split=0.2  # Split data into training (80%) and validation (20%)
)

# Prepare training and validation data
train_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=(224, 224),
    batch_size=5,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=(224, 224),
    batch_size=5,
    class_mode='binary',
    subset='validation'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Add global average pooling layer
    layers.Dense(64, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_graph.png')

# Plot training and validation loss
plt.figure()  # Create a new figure for the loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_graph.png')
