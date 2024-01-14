import time
from image_processing import classify_image, capture_image
from model import model
import schedule
import csv
from datetime import datetime


def update_csv(date, duration, csv_path='data.csv'):
    # Check if the file exists and if today's date is already recorded
    data_exists = False
    rows = []
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['date'] == date:
                    data_exists = True
                    for i in range(1, 9):
                        session_key = f'session_{i}'
                        if row[session_key] == '':
                            row[session_key] = str(duration)
                            break
                    row['total_standing'] = str(int(row['total_standing']) + duration)
                rows.append(row)

    except FileNotFoundError:
        # File doesn't exist, create it
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Update the file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['date', 'session_one', 'session_two', 'session_three', 
                      'session_four', 'session_five', 'session_six', 'session_seven', 
                      'session_eight', 'total_sessions', 'total_standing']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        # If the date is not found, add a new row
        if not data_exists:
            new_row = {field: '' for field in fieldnames}
            new_row['date'] = date
            new_row['session_one'] = str(duration)
            new_row['total_standing'] = str(duration)
            writer.writerow(new_row)

last_position, last_change_time = 'Sitting', time.time()
def job():
    global last_position, last_change_time
    capture_image()
    classification = classify_image(model, 'current_frame.jpg')
    print(classification)

    current_time = time.time()

    if classification != last_position and last_position == 'Standing':
        duration = round((current_time - last_change_time) / 60)  # Duration in minutes
        date = datetime.now().strftime("%Y-%m-%d")
        update_csv(date, duration)
        last_change_time = current_time

    last_position = classification

schedule.every(10).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)