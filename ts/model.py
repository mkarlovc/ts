import pandas
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import math
import sys

# Autoregressive (AR) model in sample
def AR(endog):
    l = len(endog)
    # Build model
    ar_model = sm.tsa.AR(endog, freq='A')
    pandas_ar_res = ar_model.fit(maxlag=None, method='mle', disp=-1)
    # Predict
    sample = pandas_ar_res.predict(start=0, end=l-1)
    return sample

# Autoregressive (AR) model predict the next value
def AR_predict_one(endog):
    l = len(endog)
    # Build model
    ar_model = sm.tsa.AR(endog, freq='A')
    pandas_ar_res = ar_model.fit(maxlag=None, method='mle', disp=-1)
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l)
    return prediction[0]

# Autoregressive (AR) model predict N next values
def AR_predict_N(endog, N):
    l = len(endog)
    # Build model
    ar_model = sm.tsa.AR(endog, freq='A')
    pandas_ar_res = ar_model.fit(maxlag=None, method='mle', disp=-1)
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l+N)
    return (prediction, range(l,l+N+1)) 

# Autoregressive Moving Average (ARMa) model
def ARMA(endog):
    # Build model
    model = sm.tsa.ARMA(endog, (2,0))
    pandas_ar_res = model.fit()
    # Predict
    sample = pandas_ar_res.predict(start=0, end=len(endog)-1)

def validation_prev(data, N):
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    for i,val in enumerate(data):
        if i<len(data)-1 and i>len(data)-1-N:
            sum += (val - data[i+1]) * (val - data[i+1])
            count += 1
            if val < min:
                min = val
            if val > max:
                max = val
    validation = {}
    validation["RMSE"] = math.sqrt(sum/count)
    validation["NRMSD"] = math.sqrt(sum/count)/(max-min)
    return validation

# Forward Chaning approcah for validating ts model
def AR_validation_fc(endog, N):
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    y = []
    x = []
    for i,val in enumerate(endog):
        if (i>len(endog)-1-N):
            pred = AR_predict_one(endog[0:i-1])
            sum += (val - pred) * (val - pred)
            count += 1
            x.append(i)
            y.append(pred)
            if val < min:
                min = val
            if val > max:
                max = val
    validation = {}
    validation["RMSE"] = math.sqrt(sum/count)
    validation["NRMSD"] = math.sqrt(sum/count)/(max-min)
    return (y, x, validation)

# Plot
def plot(title, endog, sample):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(range(0,len(endog)), endog, 'o', label="Data")
    ax.plot(range(0,len(endog)), sample, 'b-', label="Model")
    ax.legend(loc="best");
    plt.show()

def plot_pred(title, endog, sample, pred, predX):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(range(0,len(endog)), endog, 'o', label="Data")
    ax.plot(range(0,len(endog)), sample, 'b-', label="Model")
    ax.plot(range(predX[0],predX[len(predX)-1]+1), pred, 'r-', label="Predicted")
    ax.legend(loc="best");
    plt.show()
