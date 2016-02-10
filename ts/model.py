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
    return sample

def ARMA_predict_N(endog, N):
    l = len(endog)
    # Build model
    arma_model = sm.tsa.ARMA(endog, (2,0))
    pandas_ar_res = arma_model.fit()
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l+N)
    return (prediction, range(l,l+N+1))

def ARMA_predict_one(endog):
    l = len(endog)
    # Build model
    arma_model = sm.tsa.ARMA(endog, (2,0))
    pandas_arma_res = arma_model.fit()
    # Predict
    prediction = pandas_arma_res.predict(start=l, end=l)
    return prediction[0]

def validation_prev(data, N):
    out = []
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    for i,val in enumerate(data):
        if i<len(data)-1 and i>=len(data)-1-N:
            out.append(data[i])
            sum += (val - data[i+1]) * (val - data[i+1])
            count += 1
            if val < min:
                min = val
            if val > max:
                max = val
    validation = {}
    validation["RMSE"] = math.sqrt(sum/count)
    validation["NRMSD"] = math.sqrt(sum/count)/(max-min)
    return out,validation

# Forward Chaning approcah for validating ts model
def AR_fc(endog, N):
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    y = []
    x = []
    for i,val in enumerate(endog):
        if (i>len(endog)-1-N):
            pred = AR_predict_one(endog[0:i])
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

def AR_val(endog, N):
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    y = []
    x = []
    pred = AR(endog)
    for i,val in enumerate(pred):
        if (i>len(pred)-1-N):
            sum += (val - endog[i]) * (val - endog[i])
            count += 1
            x.append(i)
            y.append(val)
            if val < min:
                min = val
            if val > max:
                max = val
    validation = {}
    validation["RMSE"] = math.sqrt(sum/count)
    validation["NRMSD"] = math.sqrt(sum/count)/(max-min)
    return (y, x, validation)

def ARMA_fc(endog, N):
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    y = []
    x = []
    for i,val in enumerate(endog):
        if (i>len(endog)-1-N):
            pred = ARMA_predict_one(endog[0:i])
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

def ARMA_val(endog, N):
    sum = 0
    count = 0
    min = sys.float_info.max
    max = -sys.float_info.max
    y = []
    x = []
    pred = ARMA(endog)
    for i,val in enumerate(pred):
        if (i>len(pred)-1-N):
            sum += (val - endog[i]) * (val - endog[i])
            count += 1
            x.append(i)
            y.append(val)
            if val < min:
                min = val
            if val > max:
                max = val
    validation = {}
    validation["RMSE"] = math.sqrt(sum/count)
    validation["NRMSD"] = math.sqrt(sum/count)/(max-min)
    return (y, x, validation)

# Plot
def plot(title, endog, sample, N):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(range(0,len(endog[0:N])), endog[0:N], 'o', label="Data")
    ax.plot(range(0,len(endog[0:N])), sample[0:N], 'b-', label="Model")
    ax.legend(loc="best");
    plt.show()

def plot_pred(title, endog, sample, pred, predX):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(predX, endog[len(endog)-len(predX):len(endog)], 'o', label="Data")
    ax.plot(predX, sample[len(sample)-len(predX):len(sample)], 'b-', label="Model")
    ax.plot(predX[0:len(predX)], pred[len(pred)-len(predX):len(pred)], 'r-', label="Predicted")
    ax.legend(loc="best");
    plt.show()

def plot_pred_4(title, endog, prev, ar, arma, predX):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(predX, endog[len(endog)-len(predX):len(endog)], 'o', label="Data")
    ax.plot(predX, prev[len(prev)-len(predX):len(prev)], 'b-', label="Prev")
    ax.plot(predX, ar[len(ar)-len(predX):len(ar)], 'g-', label="AR")
    ax.plot(predX, arma[len(arma)-len(predX):len(arma)], 'r-', label="ARMA")
    ax.legend(loc="best");
    plt.show()

