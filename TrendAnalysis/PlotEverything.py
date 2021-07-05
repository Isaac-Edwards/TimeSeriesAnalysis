import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "C:\\Projects\\chaz\\SIGWATCH\\Data\\SIGWATCH_issue_data_export_REDACTED.csv"

df = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)

issue_codes = df.issue_code.unique()

ax = None
for issue in issue_codes:
    single_issue_df = df[df['issue_code'] == issue].copy()
    single_issue_df = single_issue_df.value_counts(['date', 'issue_code'], sort=False).reset_index(name='mentions')
    single_issue_df.set_index('date', inplace=True)

    single_issue_df['cumulative_mentions'] = single_issue_df['mentions'].cumsum()

    if ax is None:
        ax = single_issue_df['cumulative_mentions'].plot()
    else:
        single_issue_df['cumulative_mentions'].plot(ax=ax)

plt.show()
