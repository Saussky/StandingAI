import time
from image_processing import classify_image, capture_image
from model import model
import schedule

def job():
    capture_image()
    classification = classify_image(model, 'current_frame.jpg')
    print(classification)


schedule.every(10).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)