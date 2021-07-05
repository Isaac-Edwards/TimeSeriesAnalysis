import pandas as pd

DATA_PATH = "C:\\Projects\\chaz\\SIGWATCH\\Data\\SIGWATCH_issue_data_export_REDACTED.csv"


def read_csv(csv_path):
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


def add_cumulative_mentions(data):
    data = data.sort_values('date')
    data['cumulative_mentions'] = data.groupby('issue_code').cumcount()
    data = data.drop_duplicates(subset=['date', 'issue_code'], keep='last')
    return data
