import csv
import cv2
import datetime
from datetime import timedelta, datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import torch


def crop_image(image_array):
    """
    Crop the image to focus on the specific area of interest.
    Args:
    - image_array: The image array (NumPy array).
    Returns:
    - The cropped image array (NumPy array).
    """
    height, width = image_array.shape[:2]
    start_row, start_col = 0, width // 8
    end_row, end_col = height // 3, 7 * width // 8
    return image_array[start_row:end_row, start_col:end_col, :]


def preprocess_image(image_array, target_size=(224, 224)):
    """
    Resize image to the target size and apply MobileNetV2 preprocessing.
    Args:
    - image_array: The cropped image array (NumPy array).
    - target_size: The target size to resize the image after cropping.
    Returns:
    - A NumPy array of the preprocessed image.
    """
    image_array = image_array.astype("float32")
    resized_image = tf.image.resize(image_array, target_size)
    return preprocess_input(resized_image)


def crop_and_preprocess_image(
    image_array, crop_function, preprocess_function, target_size=(224, 224)
):
    """
    Crop and preprocess the image for MobileNetV2.
    Args:
    - image_array: The image array (NumPy array).
    - target_size: The target size to resize the image after cropping.
    Returns:
    - A NumPy array of the preprocessed image.
    """
    cropped_image = crop_function(image_array)
    preprocessed_image = preprocess_function(cropped_image, target_size)
    return preprocessed_image


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = crop_image(img_array)
    img_array = preprocess_image(img_array, target_size)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image(model, image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    print("prediction is: ", prediction)
    classification = "Standing" if prediction[0][0] > 0.5 else "Sitting"

    if 0.2 < prediction[0][0] < 0.8:
        print("Uncertain prediction, saving image for manual classification")

        file_name = os.path.basename(image_path)
        new_file_path = os.path.join("to_sort", file_name)
        shutil.copyfile(image_path, new_file_path)

    return classification


def capture_image():
    cap = cv2.VideoCapture(0)
    success, image = cap.read()

    current_datetime = datetime.now()
    date_time_str = current_datetime.strftime("%Y%m%d_%H%M")
    image_name = f"{date_time_str}.jpg"

    if success:
        cv2.imwrite(image_name, image)
    cap.release()
    cv2.destroyAllWindows()

    return image_name


def is_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def save_cropped_image(image, directory="./cropped", filename_prefix="cropped"):
    """
    Save the given PIL Image to the specified directory with a timestamped filename.
    Args:
    - image: PIL Image to be saved.
    - directory: The directory where the image will be saved.
    - filename_prefix: Prefix for the filename.
    """
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    image.save(filepath)
    print(f"Image saved to {filepath}")


def show_and_save_cropped_image(image_path, crop_function, save_dir="./cropped"):
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
    axs[0].set_title("Original Image")
    height, width = image_array.shape[:2]
    rect = patches.Rectangle(
        (width // 8, 0),
        6 * width // 8,
        height // 3,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0].add_patch(rect)
    axs[1].imshow(cropped_image_array)
    axs[1].set_title("Cropped Image")
    for ax in axs:
        ax.axis("off")
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{filename}_cropped.png")
    Image.fromarray(cropped_image_array.astype("uint8")).save(save_path)
    # plt.show()
    plt.close(fig)


# def crop_image_pil(image, save_cropped=False):
#     """
#     Crop the image to focus on the specific area of interest and optionally save it.
#     Args:
#     - image: The image as a PIL Image.
#     - save_cropped: Boolean indicating whether to save the cropped image.
#     Returns:
#     - The cropped image as a PIL Image.
#     """
#     width, height = image.size
#     start_row, start_col = 0, width // 8
#     end_row, end_col = height // 3, 7 * width // 8
#     cropped_image = image.crop((start_col, start_row, end_col, end_row))

#     if save_cropped:
#         # Ensure the ./cropped directory exists
#         os.makedirs("./cropped", exist_ok=True)
#         # Generate a timestamped filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"./cropped/cropped_{timestamp}.jpg"
#         # Save the cropped image
#         cropped_image.save(filename)
#         print(f"Cropped image saved to {filename}")

#     return cropped_image


def crop_image_pil(image, save_cropped=False, skip_crop=False):
    """
    Crop the image to keep only the top 1/3rd of the photo vertically,
    while retaining the entire width horizontally.
    Args:
    - image: The image as a PIL Image.
    - save_cropped: Save image to disk if True.
    Returns:
    - The cropped image as a PIL Image.
    """
    if skip_crop:
        return image
    
    width, height = image.size
    new_height = height // 3
    cropped_image = image.crop((0, 0, width, new_height))
    if save_cropped:
        save_cropped_image(cropped_image)
    return cropped_image


def preprocess_image_pil(image, target_size=(224, 224)):
    """
    Resize image to the target size without cropping.
    Args:
    - image: The cropped image as a PIL Image.
    - target_size: The target size to resize the image.
    Returns:
    - A PIL image of the resized image.
    """
    # Resize image without cropping, may change aspect ratio
    # image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


def crop_and_preprocess_image_pil(
    image, crop_function, preprocess_function, target_size=(224, 224)
):
    """
    Crop and preprocess the image.
    Args:
    - image: The image as a PIL Image.
    - target_size: The target size to resize the image after cropping.
    Returns:
    - A tensor of the preprocessed image.
    """
    cropped_image = crop_function(image, False, True)
    preprocessed_image = preprocess_function(cropped_image, target_size)
    return preprocessed_image


def load_and_preprocess_image_pil(image_path, target_size=(224, 224)):
    """
    Load an image file and apply preprocessing steps compatible with the trained model.
    This function assumes that the image file is a standard format that PIL can open (like JPG).
    """
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: crop_and_preprocess_image_pil(x, crop_image_pil, preprocess_image_pil, target_size=target_size)),
        ]
    )
    
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)  # Add a batch dimension


def classify_image_pytorch(model, image_path, device):
    model.eval()  # Set model to evaluation mode
    img_tensor = load_and_preprocess_image_pil(image_path)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        prediction = torch.sigmoid(
            outputs
        )
        print('predict', prediction)
        prediction = prediction.item()

    print("Prediction score:", prediction)
    classification = "Standing" if prediction > 0.5 else "Sitting"

    if 0.2 < prediction < 0.8:
        print("Uncertain prediction, saving image for manual classification")
        file_name = os.path.basename(image_path)
        new_file_path = os.path.join("to_sort", file_name)
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        shutil.copyfile(image_path, new_file_path)

    return classification
