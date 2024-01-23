import schedule
import time
from classes import JobState

job_state = JobState()
schedule.every(1).minutes.do(job_state.capture_and_process)

while True:
    schedule.run_pending()
    time.sleep(1)