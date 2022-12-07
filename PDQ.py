import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('Enrollments Forecasting - enrollments.csv')
# df = pd.read_csv('Enrollments Forecasting - enrollments overall.csv')
# df = pd.read_csv('Enrollments Forecasting - enrollments oct.csv')
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)
ts = df['enrollment count']


# Augmented Dickey-Fuller test (D) - determine stationarity of the data. Data is stationary if it does not reject the
# null hypothesis (ie the lagged level of the series does not provide relevant information in predicting the change in
# values besides the one obtained in the lagged changes. Non-stationary data rejects the null hypothesis, so requires
# differentiation.

dftest = adfuller(ts)
adf = dftest[0]
pvalue = dftest[1]
critical_value = dftest[4]['5%']

# if the pvalue is > 0.05 we can reject the null hypothesis. d will have a value of 1 (or more). if pvalue
# is < 0.05 then we cannot reject the null hypothesis. d will have a value of 0.

print(pvalue, adf, critical_value)

# calculate P value. Take a look at the first lag, if it's significantly outside of the limit, but the second isn't, go
# with 1, else keep adding the lags into the number as you go along.

plot_pacf(ts)
plt.show()

# calculate Q value. Count how many lags are outside the limit.

plot_acf(ts)
plt.show()
