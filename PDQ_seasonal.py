import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('Enrollments Forecasting - enrollments.csv')
# df = pd.read_csv('Enrollments Forecasting - enrollments overall.csv')
# df = pd.read_csv('Enrollments Forecasting - enrollments oct.csv')
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)
ts = df['enrollment count']
# ts_train = ts[:'2022-06-30']
# ts_test = ts['2022-06-30':]

result = seasonal_decompose(ts, model='multiplicative', extrapolate_trend='freq')

# result.plot()
# plt.show()

# determine D value. if stationary then 0

def check_stationarity(ts):
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    critical_value = dftest[4]['5%']
    test_statistic = dftest[0]
    # choose pvalue sensitivity of 0.05 or 0.001
    # alpha = 1e-3
    alpha = .05
    pvalue = dftest[1]
    if pvalue < alpha and test_statistic < critical_value:  # null hypothesis: x is non stationary
        print("X is stationary")
        return True
    else:
        print("X is not stationary")
        return False


ts_diff = pd.Series(ts)
d = 0
while check_stationarity(ts_diff) is False:
    ts_diff = ts_diff.diff().dropna()
    d = d + 1

seasonal = result.seasonal
check_stationarity(seasonal)
print(d)

# determine P value. count the number of lags significantly out of the limit. This can vary based on interpretation

plot_pacf(seasonal)
plt.show()

# determine Q value. count number of lags out of CI

plot_acf(seasonal)
plt.show()

