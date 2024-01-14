import tensorflow as tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image # TODO: same import? might be a bit redundant
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_path = './images'
batch_size = 5

# Initialize an image data generator with some data augmentation
train_datagen = ImageDataGenerator(
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