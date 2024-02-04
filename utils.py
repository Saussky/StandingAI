import csv
import cv2
import datetime
from datetime import timedelta
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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
    print('prediction is: ', prediction)
    classification = 'Standing' if prediction[0][0] > 0.5 else 'Sitting'
    
    if 0.2 < prediction[0][0] < 0.8:
        print('Uncertain prediction, saving image for manual classification')

        file_name = os.path.basename(image_path)
        new_file_path = os.path.join('to_sort', file_name)
        shutil.copyfile(image_path, new_file_path)
    
    return classification


def capture_image():
    cap = cv2.VideoCapture(0)
    success, image = cap.read()

    current_datetime = datetime.datetime.now()
    date_time_str = current_datetime.strftime("%Y%m%d_%H%M")
    image_name = f"{date_time_str}.jpg"

    if success:
        cv2.imwrite(image_name, image)
    cap.release()
    cv2.destroyAllWindows()

    return image_name


def update_csv(date, duration, csv_path='data.csv'):
    fieldnames = ['date', 'standing_duration', 'total_sessions', 'total_standing_minutes']

    # Initialize an empty list to store rows
    rows = []

    # Check if the file exists and if today's date is already recorded
    data_exists = False
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['date'] == date:
                    data_exists = True
                    # Update the standing duration and total standing minutes for the existing date
                    row['standing_duration'] = str(int(row['standing_duration']) + duration)
                    row['total_sessions'] = str(int(row['total_sessions']) + 1)
                    row['total_standing_minutes'] = str(int(row['total_standing_minutes']) + duration)
                rows.append(row)

    # Create file if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Update the file with modified data or add a new row
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if data_exists:
            for row in rows:
                writer.writerow(row)
        else:
            new_row = {
                'date': date,
                'standing_duration': str(duration),
                'total_sessions': '1',
                'total_standing_minutes': str(duration)
            }
            writer.writerow(new_row)


def read_csv_data(csv_path='data.csv'):
    """
    Read CSV data and return a list of dictionaries.
    """
    data = []
    if not os.path.exists(csv_path):
        return data

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def get_weekly_stats(data, week_start):
    """
    Get total standing minutes and session count for the specified week.
    """
    total_minutes = 0
    total_sessions = 0
    for row in data:
        row_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
        if week_start <= row_date < week_start + timedelta(days=7):
            total_minutes += int(row['total_standing_minutes'])
            total_sessions += int(row['total_sessions'])
    return total_minutes, total_sessions


def get_average_standing_duration(data):
    """
    Calculate the average standing duration for the specified week.
    """
    if not data:
        return 0
    total_standing_minutes = sum(int(row['total_standing_minutes']) for row in data)
    total_sessions = sum(int(row['total_sessions']) for row in data)
    average_duration = total_standing_minutes / total_sessions if total_sessions else 0
    return average_duration


def get_longest_standing_session(data):
    longest_session = max(data, key=lambda x: int(x['standing_duration'])) if data else None
    return longest_session['standing_duration'] if longest_session else 0


def get_total_standing_days(data):
    unique_days = {row['date'] for row in data}
    return len(unique_days)


def get_longest_standing_streak(data):
    if not data:
        return 0
    sorted_dates = sorted({datetime.datetime.strptime(row['date'], '%Y-%m-%d').date() for row in data})
    longest_streak = current_streak = 1
    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 1
    return longest_streak


def get_day_of_week_stats(data):
    day_stats = {day: {'total_minutes': 0, 'total_sessions': 0} for day in range(7)}  # 0: Monday, 6: Sunday
    for row in data:
        row_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
        day_of_week = row_date.weekday()
        day_stats[day_of_week]['total_minutes'] += int(row['total_standing_minutes'])
        day_stats[day_of_week]['total_sessions'] += int(row['total_sessions'])
    return day_stats


def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


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
    image_array = image_array.astype('float32')
    resized_image = tf.image.resize(image_array, target_size)
    return preprocess_input(resized_image)


def crop_and_preprocess_image(image_array, crop_function, preprocess_function, target_size=(224, 224)):
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