import os
import time
from datetime import datetime
from utils import capture_image, classify_image, update_csv
from model import model


class JobState:
    def __init__(self):
        self.last_position = 'Sitting'
        self.last_change_time = time.time()

    def job(self):
        img_name = capture_image()
        classification = classify_image(model, img_name)
        print(classification)

        current_time = time.time()

        if classification != self.last_position and self.last_position == 'Standing':
            duration = round((current_time - self.last_change_time) / 60)
            date = datetime.now().strftime("%Y-%m-%d")
            update_csv(date, duration)
            self.last_change_time = current_time

        self.last_position = classification
        os.remove(img_name)

