import datetime
import tkinter as tk
from datetime import timedelta

from utils import get_weekly_stats, read_csv_data

def create_gui():
    window = tk.Tk()
    window.title("Standing Tracker")

    data = read_csv_data('data.csv')
    today = datetime.date.today()
    today_data = next((row for row in data if row['date'] == str(today)), None)

    # Get this week's and last week's stats
    this_week_start = today - timedelta(days=today.weekday())
    last_week_start = this_week_start - timedelta(days=7)

    this_week_minutes, this_week_sessions = get_weekly_stats(data, this_week_start)
    last_week_minutes, _ = get_weekly_stats(data, last_week_start)
    weekly_difference = this_week_minutes - last_week_minutes

    # Today's stats
    today_label = tk.Label(window, text=f"Today's Total Standing Duration: {today_data['total_standing_minutes'] if today_data else '0'} minutes")
    today_label.pack()

    today_sessions_label = tk.Label(window, text=f"Today's Session Count: {today_data['total_sessions'] if today_data else '0'}")
    today_sessions_label.pack()

    # This week's stats
    week_label = tk.Label(window, text=f"This Week's Total Standing Duration: {this_week_minutes} minutes")
    week_label.pack()

    week_sessions_label = tk.Label(window, text=f"This Week's Session Count: {this_week_sessions}")
    week_sessions_label.pack()

    # Last week's comparison
    comparison_label = tk.Label(window, text=f"Difference from Last Week: {weekly_difference} minutes")
    comparison_label.pack()

    window.mainloop()

