import pandas
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import math
import sys
from sklearn.svm import SVR
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller

# stationarity

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

# standard ML models

def SVR(features, labels):
    svr_lin = SVR(kernel='linear', C=1e3)

def RF(features, labels):
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(features, labels)
    return rf

def SVM(features, labels):
    clf = svm.SVC()
    clf.fit(features, labels)
    return clf

def predict(feature_tuple, model):
    return model.predict(feature_tuple)

def RF_fc(features, labels):
    minval = sys.float_info.max
    maxval = -sys.float_info.max
    y = []
    y_true = []
    x = []
    for i,f in enumerate(features):
        if i>0 and i < len(features):
            real_val = labels[i]
            test_tuple = features[i]
            clf = RandomForestClassifier(n_estimators = 100)
            clf.fit(features[0:i], labels[0:i])
            pred = clf.predict(test_tuple)[0]
            x.append(i)
            y.append(pred)
            y_true.append(real_val)
    
    validation = val_metric(y, y_true)
    return (y,x,validation, clf)

# validation metric

def val_metric(y_pred, y_true):
    y_true_mean = sum(y_true)/len(y_true)
    SE_line = 0
    SE_y_mean = 0
    minval = sys.float_info.max
    maxval = -sys.float_info.max
    for i,p in enumerate(y_pred):
        SE_line += ((p - y_true[i]) * (p - y_true[i]))
        SE_y_mean += ((y_true[i] - y_true_mean) * (y_true[i] - y_true_mean))
        if y_true[i] < minval:
            minval = y_true[i]
        if y_true[i] > maxval:
            maxval = y_true[i]

    R2 = 1 - (SE_line/SE_y_mean)
    RSS = SE_line
    MSE = SE_line/len(y_pred)
    RMSE = math.sqrt(SE_line/len(y_pred))
    NRMSD = math.sqrt(SE_line/len(y_pred))/(maxval-minval)
    return {"R2": R2, "RMSE":RMSE, "NRMSD":NRMSD, "MSE":MSE, "RSS":RSS}

def SVM_fc(features, labels):
    for i,f in enumerate(features):
        if i>5 and i < len(features)-1:
            real_val = labels[i+1]
            test_tuple = features[i+1]
            clf = svm.SVC()
            clf.fit(features[0:i], labels[0:i])
            print test_tuple, clf.predict(test_tuple), real_val

# Autoregressive (AR) model in sample
def AR(endog):
    l = len(endog)
    # Build model
    model = sm.tsa.AR(endog, freq='M')
    pandas_ar_res = model.fit(maxlag=None, method='mle', disp=-1)
    # Predict
    sample = pandas_ar_res.predict(start=0, end=l-1)
    validation = val_metric(sample,endog)
    return sample,model,validation

# Autoregressive (AR) model predict the next value
def AR_predict_one(endog):
    l = len(endog)
    # Build model
    ar_model = sm.tsa.AR(endog, freq='M')
    pandas_ar_res = ar_model.fit(maxlag=None, method='mle', disp=-1)
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l)
    return prediction[0]

# Autoregressive (AR) model predict N next values
def AR_predict_N(endog, N):
    l = len(endog)
    # Build model
    ar_model = sm.tsa.AR(endog, freq='M')
    pandas_ar_res = ar_model.fit(maxlag=None, method='mle', disp=-1)
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l+N)
    return (prediction, range(l,l+N+1)) 

# Autoregressive Moving Average (ARMa) model
def ARMA(endog, p, q):
    # Build model
    model = sm.tsa.ARMA(endog, (p,q), freq='M')
    pandas_ar_res = model.fit()
    # Predict
    sample = pandas_ar_res.predict(start=0, end=len(endog)-1)
    validation = val_metric(sample,endog)
    return sample,model,validation

def ARMA_predict_N(endog, N, p, q):
    l = len(endog)
    # Build model
    arma_model = sm.tsa.ARMA(endog, (p,q), freq='M')
    pandas_ar_res = arma_model.fit()
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l+N)
    return (prediction, range(l,l+N+1))

def ARMA_predict_one(endog, p, q):
    l = len(endog)
    # Build model
    arma_model = sm.tsa.ARMA(endog, (p,q), freq='M')
    pandas_arma_res = arma_model.fit()
    # Predict
    prediction = pandas_arma_res.predict(start=l, end=l)
    return prediction[0]

# Autoregressive Integrated Moving Average (ARIMA) model
def ARIMA(endog, p, d, q):
    # Build model
    model = sm.tsa.ARIMA(endog, (p,d,q), freq='M')
    pandas_ar_res = model.fit()
    # Predict
    sample = pandas_ar_res.predict(start=0, end=len(endog)-1)
    validation = val_metric(sample,endog)
    return sample,model,validation

def ARIMA_predict_N(endog, N, p, d, q):
    l = len(endog)
    # Build model
    arma_model = sm.tsa.ARIMA(endog, (p,d,q), freq='M')
    pandas_ar_res = arma_model.fit()
    # Predict
    prediction = pandas_ar_res.predict(start=l, end=l+N)
    return (prediction, range(l,l+N+1))

def ARIMA_predict_one(endog, p, d, q):
    l = len(endog)
    # Build model
    arma_model = sm.tsa.ARIMA(endog, (p,d,q), freq='M')
    pandas_arma_res = arma_model.fit()
    # Predict
    prediction = pandas_arma_res.predict(start=l, end=l)
    return prediction[0]

def ARIMA_fc(endog, N, p, d, q):
    print "ARIMA",len(endog)
    y = []
    y_true = []
    x = []
    len(endog)
    for i,val in enumerate(endog):
        if (i>len(endog)-1-N) and i>p and i>q and i>d:
            pred = ARIMA_predict_one(endog[0:i], p, d, q)
            x.append(i)
            y.append(pred)
            y_true.append(val)
    validation = val_metric(y, y_true)
    return (y, x, validation)

def validation_prev(data, N):
    out = []
    y = []
    x = []
    y_true = []
    for i,val in enumerate(data):
        if i<len(data)-1 and i>=len(data)-1-N:
            out.append(data[i])
            y.append(data[i+1])
            y_true.append(val)
    validation = val_metric(y, y_true)
    return out,validation

# Forward Chaning approcah for validating ts model
def AR_fc(endog, N):
    y = []
    x = []
    y_true = []
    for i,val in enumerate(endog):
        if (i>len(endog)-1-N):
            pred = AR_predict_one(endog[0:i])
            x.append(i)
            y.append(pred)
            y_true.append(val)
    
    validation = val_metric(y, y_true)
    return (y, x, validation)

def feature_set(endog, win_size, br):
    win = []
    sumv = 0
    avg = 0
    maxv = -1
    minv = sys.maxint
    features = []
    labels = []
    for i,val in enumerate(endog):
        win.append(val)
        sumv += val
        avg = sumv/win_size
        if i>len(endog)-1-br-win_size and i>win_size and i < len(endog)-1:
            label = endog[i+1]
            sumv -= win[0]
            del win[0]
            maxv = max(win)
            minv = min(win)
            f = [avg]
            features.append(f+win)
            labels.append(endog[i+1])
        else:
            if i>win_size and i < len(endog)-1:
                label = endog[i+1]
                sumv -= win[0]
                del win[0]
                maxv = max(win)
                minv = min(win)
                f = [avg]
                features.append(f+win)
                labels.append(endog[i+1])
    Y = np.array(features)
    #Y = preprocessing.normalize(np.array(features))
    return (Y,np.array(labels))

def AR_val(endog, N):
    y = []
    y_true = []
    x = []
    pred = AR(endog)
    for i,val in enumerate(pred):
        if (i>len(pred)-1-N):
            x.append(i)
            y.append(val)
            y_true.append(endog[i])
    validation = val_metric(y, y_true)
    return (y, x, validation)

def ARMA_fc(endog, N, p, q):
    y = []
    y_true = []
    x = []
    for i,val in enumerate(endog):
        if (i>len(endog)-1-N) and i>p and i>q:
            pred = ARMA_predict_one(endog[0:i], p, q)
            x.append(i)
            y.append(pred)
            y_true.append(val)
    validation = val_metric(y, y_true)
    return (y, x, validation)

def ARMA_val(endog, N):
    y = []
    y_true = []
    x = []
    pred = ARMA(endog)
    for i,val in enumerate(pred):
        if (i>len(pred)-1-N):
            x.append(i)
            y.append(val)
            y_true.append(endog[i])
    validation = val_metric(y, y_true)
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

def plot_pred1(title, endog, pred, predX, val):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(predX, endog[len(endog)-len(predX):len(endog)], 'o', label="Data")
    ax.plot(predX, pred[len(pred)-len(predX):len(pred)], 'r-', label="Predicted ("+str(val)+")")
    ax.legend(loc="best");
    plt.show()

def plot_pred_4(title, endog, prev, ar, arma, predX, prev_rmse, ar_rmse, arma_rmse):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(predX, endog[len(endog)-len(predX):len(endog)], 'o', label="Data")
    ax.plot(predX, prev[len(prev)-len(predX):len(prev)], 'b-', label="Prev("+str(round(prev_rmse,3))+")")
    ax.plot(predX, ar[len(ar)-len(predX):len(ar)], 'g-', label="AR("+str(round(ar_rmse,3))+")")
    ax.plot(predX, arma[len(arma)-len(predX):len(arma)], 'r-', label="ARMA("+str(round(arma_rmse,3))+")")
    ax.legend(loc="best");
    plt.show()

