import os
import shutil
import numpy as np
import cv2
import csv
from tensorflow.keras.preprocessing import image
import datetime

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

def classify_images(model, directory):
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, file_name)
            img = load_and_preprocess_image(file_path)

            # Predict
            prediction = model.predict(img)
            classification = 'Standing' if prediction[0][0] > 0.5 else 'Sitting'

            # Print the result
            print(f"Image: {file_name} | Prediction: {classification} ({prediction[0][0]})")

def classify_image(model, image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    print('prediction is: ', prediction)
    classification = 'Standing' if prediction[0][0] > 0.5 else 'Sitting'
    
    if 0.3 < prediction[0][0] < 0.7:
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
    fieldnames = ['date', 'standing_duration_1', 'standing_duration_2', 'standing_duration_3', 
                  'standing_duration_4', 'standing_duration_5', 'total_sessions', 'total_standing_minutes']

    # Check if the file exists and if today's date is already recorded
    data_exists = False
    rows = []
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['date'] == date:
                    data_exists = True
                    session_count = 0
                    for i in range(1, 6):
                        session_key = f'standing_duration_{i}'
                        if row[session_key] == '':
                            row[session_key] = str(duration)
                            break
                        session_count += 1
                    row['total_sessions'] = str(session_count)
                    row['total_standing_minutes'] = str(int(row.get('total_standing_minutes', '0')) + duration)
                rows.append(row)

    except FileNotFoundError:
        # File doesn't exist, create it
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Update the file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        # If the date is not found, add a new row
        if not data_exists:
            new_row = {field: '' for field in fieldnames}
            new_row['date'] = date
            new_row['standing_duration_1'] = str(duration)
            new_row['total_sessions'] = '1'
            new_row['total_standing_minutes'] = str(duration)
            writer.writerow(new_row)
