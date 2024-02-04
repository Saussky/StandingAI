import schedule
import time
from classes import JobState
from gui import create_gui

job_state = JobState()
schedule.every(10).minutes.do(job_state.capture_and_process)

while True:
    schedule.run_pending()
    time.sleep(1)