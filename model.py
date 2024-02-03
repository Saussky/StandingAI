from matplotlib import patches
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

from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



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
    start_col = width // 8  # Start column at 1/8th of the width from the left
    end_row = height // 3   # End row at 1/3rd of the height from the top
    end_col = 7 * width // 8  # End column at 7/8th of the width from the left

    # Ensure the coordinates are within the image dimensions
    start_col = max(0, start_col)
    start_row = max(0, start_row)
    end_col = min(width, end_col)
    end_row = min(height, end_row)
    
    cropped_image = image_array[start_row:end_row, start_col:end_col, :]
    
    # Convert the cropped image to float32 to prepare for resizing and preprocessing
    cropped_image = cropped_image.astype('float32')

    # Resize the cropped image to the target size
    resized_image = tf.image.resize(cropped_image, target_size)

    # Apply MobileNetV2 preprocessing
    preprocessed_image = preprocess_input(resized_image)
    
    return preprocessed_image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def crop_image_for_visualization(image_array):
    """
    Crop the image to focus on the specific area of interest.

    Args:
    - image_array: The image array (NumPy array).

    Returns:
    - The cropped image array (NumPy array).
    """
    height, width = image_array.shape[:2]
    start_row = 0
    start_col = width // 8
    end_row = height // 3
    end_col = 7 * width // 8
    
    # Ensure the coordinates are within the image dimensions
    start_col = max(0, start_col)
    start_row = max(0, start_row)
    end_col = min(width, end_col)
    end_row = min(height, end_row)
    
    cropped_image = image_array[start_row:end_row, start_col:end_col, :]
    return cropped_image

def show_cropped_images(image_paths, crop_function):
    """
    Displays the original and cropped images side by side without preprocessing.

    Args:
    - image_paths: A list of file paths to the images.
    - crop_function: A function that crops the images.
    """
    fig, axs = plt.subplots(2, len(image_paths), figsize=(12, 6))
    
    for i, image_path in enumerate(image_paths):
        # Load the image
        image = Image.open(image_path)
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        # Crop the image
        cropped_image_array = crop_function(image_array)
        
        # Show original image
        axs[0, i].imshow(image_array)
        axs[0, i].set_title('Original Image')
        
        # Highlight the crop area on the original image
        start_col = width // 8
        end_col = 7 * width // 8
        start_row = 0
        end_row = height // 3
        rect = patches.Rectangle((start_col, start_row), end_col - start_col, end_row - start_row, 
                                 linewidth=1, edgecolor='r', facecolor='none')
        axs[0, i].add_patch(rect)
        
        # Show cropped image
        axs[1, i].imshow(cropped_image_array)
        axs[1, i].set_title('Cropped Image')
        
        # Remove axis ticks
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
image_paths = ['./images/standing/standing18.jpg', './images/standing/standing19.jpg']
# show_cropped_images(image_paths, crop_image_for_visualization)


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
