from TrendAnalysis.data_utils import DATA_PATH, read_csv, add_cumulative_mentions
import matplotlib.pyplot as plt


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
