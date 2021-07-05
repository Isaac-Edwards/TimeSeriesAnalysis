import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "C:\\Projects\\chaz\\SIGWATCH\\Data\\SIGWATCH_issue_data_export_REDACTED.csv"


def read_csv(csv_path):
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


def add_cumulative_mentions(data):
    data = data.sort_values('date')
    data['cumulative_mentions'] = data.groupby('issue_code').cumcount()
    data = data.drop_duplicates(subset=['date', 'issue_code'], keep='last')
    return data


def plot_all_issues(data):
    issue_codes = data['issue_code'].unique()

    data.set_index('date', inplace=True)
    ax = None
    for issue in issue_codes:
        single_issue_df = data[data['issue_code'] == issue]
        if ax is None:
            ax = single_issue_df['cumulative_mentions'].plot()
        else:
            single_issue_df['cumulative_mentions'].plot(ax=ax)

    plt.show()


def main():
    df = read_csv(DATA_PATH)
    df = add_cumulative_mentions(df)
    plot_all_issues(df)


if __name__ == "__main__":
    main()
