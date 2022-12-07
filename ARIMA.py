import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# choose file to read

# file_input = input("Enter file name: ")

# read csv

df = pd.read_csv('Enrollments Forecasting - enrollments.csv')
df.index = pd.to_datetime(df['month'], format='%Y-%m-%d')
ts = df['enrollment count']

del df['month']

# uncomment to test model. 80/20 split is standard.

train = df[df.index < pd.to_datetime("2021-12-31", format='%Y-%m-%d')]
test = df[df.index > pd.to_datetime("2021-12-31", format='%Y-%m-%d')]

y = train['enrollment count']

# run the AD-Fuller test to determine d

def check_stationarity(ts):
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    critical_value = dftest[4]['5%']
    test_statistic = dftest[0]
    # adjust alpha if you want to see different pvalue sensitivity
    alpha = .01
    pvalue = dftest[1]
    if pvalue < alpha and test_statistic < critical_value:  # null hypothesis: x is non stationary
        return True
    else:
        return False


ts_diff = pd.Series(ts)
d = 0
while check_stationarity(ts_diff) is False:
    ts_diff = ts_diff.diff().dropna()
    d = d + 1

# fit the SARIMA model
# see "PDQ.py" and "PDQ_seasonal.py" to see how I got these values

p = 1
q = 0
P = 8
Q = 1
time_interval = 12

SARIMAXmodel = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, d, Q, time_interval))
SARIMAXmodel = SARIMAXmodel.fit()

# test the SARIMA model

SARIMAXmodel_test = SARIMAX(y, order=(p, d, q), seasonal_order=(P, d, Q, time_interval))
SARIMAXmodel_test = SARIMAXmodel_test.fit()
y_pred_test = SARIMAXmodel_test.get_forecast(len(test.index))
y_pred_df_test = y_pred_test.conf_int(alpha=0.05)
y_pred_df_test["Predictions"] = SARIMAXmodel_test.predict(start=y_pred_df_test.index[0], end=y_pred_df_test.index[-1])
sarima_rmse = np.sqrt(mean_squared_error(test['enrollment count'].values, y_pred_df_test["Predictions"]))
sarima_rmse_round = '%.2f' % sarima_rmse

# predict SARIMA

y_pred = SARIMAXmodel.get_forecast(steps=6)
y_pred_df = y_pred.conf_int(alpha=0.05)
y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
# y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"]

# plot the predictions

plt.plot(ts, color="black", label='Actuals')
# plt.plot(y_pred_out_arima, color='Green', label='ARIMA Predictions')
plt.plot(y_pred_out, color='Blue', label='SARIMA Predictions')
plt.ylabel('Enrollments')
plt.xlabel('Instance Start')
plt.xticks(rotation=45)
# plt.title(f"Inventory Forecast, RMSE: {sarima_rmse_round}")
plt.title("Inventory Forecast")
plt.legend()
plt.show()

print(y_pred_df)
print(sarima_rmse_round)


