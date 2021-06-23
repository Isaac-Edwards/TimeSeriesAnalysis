import pmdarima as pmd
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

sarima = pmd.auto_arima(data,
                        start_p=1,
                        start_q=1,
                        test='adf',
                        max_p=3,
                        max_q=3,
                        m=12,
                        start_P=0,
                        seasonal=True,
                        d=None,
                        D=1,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True
                        )

print(sarima.summary())

future_periods = 24
fitted, confidence_interval = sarima.predict(n_periods=future_periods, return_conf_int=True)
forecast_index = pd.date_range(data.index[-1], periods=future_periods, freq='MS')

fitted_series = pd.Series(fitted, index=forecast_index)
lower_series = pd.Series(confidence_interval[:, 0], index=forecast_index)
upper_series = pd.Series(confidence_interval[:, 1], index=forecast_index)

plt.plot(data)
plt.plot(fitted_series)
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k',
                 alpha=0.15)
plt.title("SARIMA - Final Forecast of Australian Diabetic Drug Sales")
plt.show()
