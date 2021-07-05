import pandas as pd
from TrendAnalysis.data_utils import DATA_PATH, read_csv, add_cumulative_mentions


def crop_to_last_6_months(data):
    today = pd.to_datetime("today")
    six_months_ago = today + pd.DateOffset(months=-6)
    return data.loc[data['date'] >= six_months_ago]


def main():
    df = read_csv(DATA_PATH)
    df = add_cumulative_mentions(df)
    last_6_months = crop_to_last_6_months(df)


if __name__ == "__main__":
    main()
