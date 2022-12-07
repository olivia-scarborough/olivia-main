import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

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

# rank AIC and BIC

order_aic_bic = []

# Loop over p values from 0-4
for p in range(5):
    # Loop over q values from 0-4
    for q in range(5):

        try:
            # create and fit ARMA(p,q) model
            model = sm.tsa.statespace.SARIMAX(ts, order=(p, 1, q))
            results = model.fit()

            # Print order and results
            order_aic_bic.append((p, q, results.aic, results.bic))
        except:
            print(p, q, None, None)

# Make DataFrame of model order and AIC/BIC scores
order_df = pd.DataFrame(order_aic_bic, columns=['p', 'q', 'aic', 'bic'])

# lets sort them by AIC and BIC

# Sort by AIC
print("Models sorted by AIC ")
print("\n")
print(order_df.sort_values('aic').reset_index(drop=True))

# Sort by BIC
print("Models sorted by BIC ")
print("\n")
print(order_df.sort_values('bic').reset_index(drop=True))