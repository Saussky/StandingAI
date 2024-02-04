import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import crop_and_preprocess_image, crop_image, is_image_file, preprocess_image

image_path = './images'
base_dir = './images'
sitting_dir = os.path.join(base_dir, 'sitting')
standing_dir = os.path.join(base_dir, 'standing')

# Check if the base directory exists and contains the expected subdirectories
if not os.path.isdir(base_dir) or not os.path.isdir(sitting_dir) or not os.path.isdir(standing_dir):
    raise Exception(f"Expected directory structure not found in {base_dir}")


def show_and_save_cropped_image(image_path, crop_function, save_dir='./cropped'):
    """
    Displays the original and cropped image side by side.
    Saves the cropped images to the specified save_dir with unique filenames.
    Args:
    - image_path: The file path to the image.
    - crop_function: A function that crops the images.
    - save_dir: Directory to save the cropped images.
    """
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path)
    image_array = np.array(image)
    cropped_image_array = crop_function(image_array)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image_array)
    axs[0].set_title('Original Image')
    height, width = image_array.shape[:2]
    rect = patches.Rectangle((width // 8, 0), 6 * width // 8, height // 3, linewidth=1, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    axs[1].imshow(cropped_image_array)
    axs[1].set_title('Cropped Image')
    for ax in axs:
        ax.axis('off')
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f'{filename}_cropped.png')
    Image.fromarray(cropped_image_array.astype('uint8')).save(save_path)
    # plt.show()
    plt.close(fig)

# Loop over images in the sitting and standing directories and process them
# sitting_dir, standing_dir = './images/sitting', './images/standing'
# sitting_image_paths = [os.path.join(sitting_dir, f) for f in os.listdir(sitting_dir) if is_image_file(f)]
# standing_image_paths = [os.path.join(standing_dir, f) for f in os.listdir(standing_dir) if is_image_file(f)]
# for image_path in sitting_image_paths + standing_image_paths:
#     show_and_save_cropped_image(image_path, crop_image)

# Prepare the data augmentation generator
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: crop_and_preprocess_image(x, crop_image, preprocess_image, target_size=(224, 224)),
    # rescale=1./255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    fill_mode='nearest',
    validation_split=0.2  # Split data into 80% training and 20% validation
)


# Prepare training and validation data
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=5,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
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
