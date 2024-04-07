import csv
import datetime
import os


def update_csv(date, duration, csv_path="data.csv"):
    fieldnames = [
        "date",
        "standing_duration",
        "total_sessions",
        "total_standing_minutes",
    ]

    # Initialize an empty list to store rows
    rows = []

    # Check if the file exists and if today's date is already recorded
    data_exists = False
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["date"] == date:
                    data_exists = True
                    # Update the standing duration and total standing minutes for the existing date
                    row["standing_duration"] = str(
                        int(row["standing_duration"]) + duration
                    )
                    row["total_sessions"] = str(int(row["total_sessions"]) + 1)
                    row["total_standing_minutes"] = str(
                        int(row["total_standing_minutes"]) + duration
                    )
                rows.append(row)

    # Create file if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Update the file with modified data or add a new row
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if data_exists:
            for row in rows:
                writer.writerow(row)
        else:
            new_row = {
                "date": date,
                "standing_duration": str(duration),
                "total_sessions": "1",
                "total_standing_minutes": str(duration),
            }
            writer.writerow(new_row)


def read_csv_data(csv_path="data.csv"):
    """
    Read CSV data and return a list of dictionaries.
    """
    data = []
    if not os.path.exists(csv_path):
        return data

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def get_weekly_stats(data, week_start):
    """
    Get total standing minutes and session count for the specified week.
    """
    total_minutes = 0
    total_sessions = 0
    for row in data:
        row_date = datetime.datetime.strptime(row["date"], "%Y-%m-%d").date()
        if week_start <= row_date < week_start + datetime.timedelta(days=7):
            total_minutes += int(row["total_standing_minutes"])
            total_sessions += int(row["total_sessions"])
    return total_minutes, total_sessions


def get_average_standing_duration(data):
    """
    Calculate the average standing duration for the specified week.
    """
    if not data:
        return 0
    total_standing_minutes = sum(int(row["total_standing_minutes"]) for row in data)
    total_sessions = sum(int(row["total_sessions"]) for row in data)
    average_duration = total_standing_minutes / total_sessions if total_sessions else 0
    return average_duration


def get_longest_standing_session(data):
    longest_session = (
        max(data, key=lambda x: int(x["standing_duration"])) if data else None
    )
    return longest_session["standing_duration"] if longest_session else 0


def get_total_standing_days(data):
    unique_days = {row["date"] for row in data}
    return len(unique_days)


def get_longest_standing_streak(data):
    if not data:
        return 0
    sorted_dates = sorted(
        {datetime.datetime.strptime(row["date"], "%Y-%m-%d").date() for row in data}
    )
    longest_streak = current_streak = 1
    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 1
    return longest_streak


def get_day_of_week_stats(data):
    day_stats = {
        day: {"total_minutes": 0, "total_sessions": 0} for day in range(7)
    }  # 0: Monday, 6: Sunday
    for row in data:
        row_date = datetime.datetime.strptime(row["date"], "%Y-%m-%d").date()
        day_of_week = row_date.weekday()
        day_stats[day_of_week]["total_minutes"] += int(row["total_standing_minutes"])
        day_stats[day_of_week]["total_sessions"] += int(row["total_sessions"])
    return day_stats
