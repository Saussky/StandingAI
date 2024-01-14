import schedule
import time
from utils import classify_image, capture_image, update_csv
from classes import JobState

job_state = JobState()
schedule.every(10).seconds.do(job_state.job)

while True:
    schedule.run_pending()
    time.sleep(1)