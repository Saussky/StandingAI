import sys
import datetime
from datetime import timedelta
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import get_average_standing_duration, get_day_of_week_stats, get_longest_standing_session, get_longest_standing_streak, get_total_standing_days, get_weekly_stats, read_csv_data

def create_heading_label(text):
    label = QLabel(text)
    label.setFont(QFont('Arial', 16, QFont.Bold))
    label.setStyleSheet("color: #2F4F4F; margin-top: 20px; margin-bottom: 5px;")
    return label

def create_stat_label(text):
    label = QLabel(text)
    label.setStyleSheet("background-color: #F0FFFF; color: black; padding: 5px; border-radius: 5px; margin: 2px;")
    return label

def create_gui():
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Standing Tracker")
    window.setStyleSheet("background-color: #E0EEEE; padding: 10px;")

    data = read_csv_data('data.csv')
    today = datetime.date.today()
    today_data = next((row for row in data if row['date'] == str(today)), None)

    # Get this week's and last week's stats
    this_week_start = today - timedelta(days=today.weekday())
    last_week_start = this_week_start - timedelta(days=7)

    this_week_minutes, this_week_sessions = get_weekly_stats(data, this_week_start)
    last_week_minutes, _ = get_weekly_stats(data, last_week_start)
    weekly_difference = this_week_minutes - last_week_minutes

    layout = QVBoxLayout()

    # Apply styled headings and stats
    layout.addWidget(create_heading_label("Today's Statistics"))
    layout.addWidget(create_stat_label(f"Today's Total Standing Duration: {today_data['total_standing_minutes'] if today_data else '0'} minutes"))
    layout.addWidget(create_stat_label(f"Today's Session Count: {today_data['total_sessions'] if today_data else '0'}"))

    layout.addWidget(create_heading_label("Weekly Summary"))
    layout.addWidget(create_stat_label(f"This Week's Total Standing Duration: {this_week_minutes} minutes"))
    layout.addWidget(create_stat_label(f"Difference from Last Week: {weekly_difference} minutes"))

    # Additional statistics with styled labels
    layout.addWidget(create_heading_label("Additional Insights"))
    layout.addWidget(create_stat_label(f"Average Standing Duration per Session: {get_average_standing_duration(data)} minutes"))
    layout.addWidget(create_stat_label(f"Longest Standing Session: {get_longest_standing_session(data)} minutes"))
    layout.addWidget(create_stat_label(f"Total Standing Days: {get_total_standing_days(data)}"))
    layout.addWidget(create_stat_label(f"Longest Standing Streak: {get_longest_standing_streak(data)} days"))

    # Day of Week Stats with Styling
    layout.addWidget(create_heading_label("Day of Week Analysis"))
    day_of_week_stats_data = get_day_of_week_stats(data)

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    total_minutes = [day_of_week_stats_data[day]['total_minutes'] for day in range(7)]

    # Create a matplotlib figure and canvas
    fig = Figure(figsize=(5, 8), dpi=100)  # Increase the height of the figure
    canvas = FigureCanvas(fig)
    canvas.setMinimumSize(600, 400)  # Set the minimum width and height for the canvas

    # Plot the pie chart on the figure
    ax = fig.add_subplot(111)
    ax.pie(total_minutes, labels=day_names, autopct='%1.1f%%')
    ax.set_title('Day of Week Analysis')

    # Add the canvas to the layout with increased stretch factor
    layout.addWidget(canvas, stretch=1)

    window.setLayout(layout)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    create_gui()

create_gui()