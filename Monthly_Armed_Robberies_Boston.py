import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt, log, exp
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox

#read data from csv
df= pd.read_csv('Robberies.csv', header=0, index_col=0, parse_dates=True)

df_series = df.squeeze()
print(df_series)

#prepare data
#evaluate a persistence(naive) model - baseline model - just predicting the next step
X = df_series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]

#walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' %(yhat , obs))

rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE of baseline model: %.3f' %rmse)

# summary statistics of time series
print(df_series.describe())

#Line plots of data
df_series.plot()
plt.show()

# Density Plots
plt.figure(1)
plt.subplot(211)
df_series.hist()

plt.subplot(212)
df_series.plot(kind='kde')
plt.show()

#create a differenced time series
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i-1]
        diff.append(value)
    return Series(diff)

#differenced data
stationary = difference(X)
stationary.index = df_series.index[1:]

#check if stationary
result = adfuller(stationary)
print(result)
print('ADF statistics: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s:  %.3f'  % (key,value))
stationary.to_csv('stationary.csv', header=False)

#obs: atleast one differencing of data is required based on the adfuller test

#ACF PACF plots of Time Series to get p and q values
plt.figure()
plt.subplot(211)

plot_acf(df_series, lags=50, ax=plt.gca())
plt.subplot(212)
plot_pacf(df_series, lags=50, ax=plt.gca())

plt.show()

# manual ARIMA with p and q values based on plots
#prerpare data
#walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    model =ARIMA(history, order=(0,1,2))
    model_fit = model.fit()
    yhat=model_fit.forecast()[0]
    predictions.append(yhat)
    #observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' %(yhat , obs))

#report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('Manual ARIMA RMSE: %.3f' %rmse)


# grid search ARIMA parameters for time series
def evaluate_arima_model(X, arima_order):
    X = X.astype('float32')
    #print('X', X)
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    #print(len(train), len(test))
    history = [x for x in train]
    predictions =  list()
    for t in range(len(test)):
        #print('t',t)
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        #print('yhat',yhat)
        predictions.append(yhat)
        history.append(test[t])
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

def evaluate_models(dataset, p_values, d_values, q_values):
    #print(p_values, d_values, q_values)
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    #print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA model using grid search%s RMSE=%.3f' % (best_cfg, best_score))


p_values = range(0,4)
d_values = range(0,2)
q_values = range(0,4)
warnings.filterwarnings("ignore")
evaluate_models(df_series.values, p_values, d_values, q_values)

#Now we review residual errors of a single ARIMA model
#model
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
# predict
    model = ARIMA(history, order=(0,1,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = pd.DataFrame(residuals)
# plotting the errors
plt.figure()
plt.subplot(211)
residuals.hist(ax=plt.gca())
plt.subplot(212)
residuals.plot(kind='kde', ax=plt.gca())
plt.show()


# ACF and PACF of residual Plots
plt.figure()
plt.subplot(211)
plot_acf(residuals, lags=25, ax=plt.gca())
plt.subplot(212)
plot_pacf(residuals, lags=25, ax=plt.gca())
plt.show()


# plot of box-cox transformed 

transformed, lam = boxcox(X)
print('Lambda: %f' % lam)
plt.figure(1)
# line plot
plt.subplot(311)
plt.plot(transformed)
# histogram
plt.subplot(312)
plt.hist(transformed)
# q-q plot
plt.subplot(313)
qqplot(transformed, line='r', ax=plt.gca())
plt.show()

# invert box-cox transform
def boxcox_inverse(value, lam):
    if lam == 0:
        return exp(value)
    return exp(log(lam * value + 1) / lam)

# using ARIMA model with Box Cox transofrmation
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # transform
    transformed, lam = boxcox(history)
    if lam < -5:
        transformed, lam = history, 1
    # predict
    model = ARIMA(transformed, order=(0,1,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    yhat = boxcox_inverse(yhat, lam)
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE of ARIMA with Boxcox transformation: %.3f' % rmse)

#Final Model
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(0,1,2))
model_fit = model.fit()
# save model
model_fit.save('robberymodel.pkl')
np.save('model_lambda.npy', [lam])

# Validate Model
# split into a training and validation dataset
from pandas import read_csv
series = read_csv('robberies.csv', header=0, index_col=0, parse_dates=True)
series = series.squeeze()
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)

# load and prepare datasets
dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
dataset = dataset.squeeze()
X = dataset.values.astype('float32')
history = [x for x in X]
validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True)
validation = validation.squeeze()
y = validation.values.astype('float32')

# load model
model_fit = ARIMAResults.load('robberymodel.pkl')
lam = np.load('model_lambda.npy')

# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
    # transform
    transformed, lam = boxcox(history)
    if lam < -5:
        transformed, lam = history, 1
    # predict
    model = ARIMA(transformed, order=(0,1,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    yhat = boxcox_inverse(yhat, lam)
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
plt.plot(y)
plt.plot(predictions, color='red')
plt.show()
