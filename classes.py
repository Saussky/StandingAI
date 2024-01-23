import os
import time
from datetime import datetime
from gui import create_gui
from utils import capture_image, classify_image, update_csv
from model import model


class JobState:
    def __init__(self):
        self.last_position = 'Sitting'
        self.last_change_time = time.time()

    def capture_and_process(self):
        try:
            img_name = capture_image()
            current_time = time.time()
            classification = classify_image(model, img_name)

            print(f'You are : {classification}')

            if classification != self.last_position and self.last_position == 'Standing':
                duration = round((current_time - self.last_change_time) / 60)
                print(f'Standing duration: {duration} minutes')
                date = datetime.now().strftime("%Y-%m-%d")
                update_csv(date, duration)
                self.last_change_time = current_time

            self.last_position = classification
            os.remove(img_name)
        except Exception as err:
            print(f"An error occurred: {str(err)}")
