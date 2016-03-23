from flask import Flask, Response, request, render_template
import json
import sys
from ts import data
from ts import model
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.api import qqplot
import datetime
from statsmodels.tsa.stattools import adfuller

app = Flask(__name__, static_url_path='')
path = "/home/luis/data/mario/openedu/"

##############
# Interface
##############

@app.route('/')
def search():
    return app.send_static_file('index.html')

#############
# API
#############

def apiout(res, indicatordf, duration):
    out = {}
    out["validation"] = res[2]
    endog = []
    prediction = []
    br = 0;
    for i,idx in enumerate(indicatordf.index):
        endog.append({"count":indicatordf.ix[idx], "date":idx.strftime('%Y-%m-%d'), "i":i})
        if i >= len(indicatordf)-int(duration)+int(duration)-len(res[0]):
            prediction.append({"count": res[0][br], "date": idx.strftime('%Y-%m-%d')})
            br += 1

    out["validation"] = res[2]
    out["input"] = endog
    out["prediction"] = prediction
    return out

def apincout(res, dateout):
    pred = []
    for i,r in enumerate(res[0]):
        pred.append({"count": r, "date": dateout[i+1]})
        out = {}
        out["validation"] = res[2]
        out["prediction"] = pred
    return out

def apioutseries(erdfs):
    out = {}
    erdfs_vec = []
    for erdf in erdfs:
        erdf_vec = []
        for i,idx in enumerate(erdf.index):
            erdf_vec.append({"count": erdf.ix[idx], "date": idx.strftime('%Y-%m-%d')})
        erdfs_vec.append(erdf_vec)
    out = {"series": erdfs_vec}
    return out
 

@app.route('/api/indicator/<textA>/<textB>', methods=['GET'])
def indicator(textA, textB):   
    indicatordf = data.getIndicatorDf(textA+"/"+textB)
    output = []
    for idx in indicatordf.index:
        output.append({"count":indicatordf.ix[idx], "date":idx.strftime('%Y-%m-%d')})
    return Response(json.dumps(output), mimetype='application/json')

@app.route('/api/indicator/<textA>/<textB>/<d>', methods=['GET'])
def indicator_d(textA, textB, d):
    indicatordf = data.getIndicatorDf(textA+"/"+textB)
    output = []
    for i,idx in enumerate(indicatordf.index):
        if i > len(indicatordf)-int(d):
            output.append({"count":indicatordf.ix[idx], "date":idx.strftime('%Y-%m-%d')})
    return Response(json.dumps(output), mimetype='application/json')

@app.route('/api/er/<text>', methods=['GET'])
def er(text):
    d = data.erGetCountsConcept("http://en.wikipedia.org/wiki/"+text)["http://en.wikipedia.org/wiki/"+text]
    s = pd.Series(d)
    df = pd.DataFrame(d).set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.groupby(pd.TimeGrouper("M")).sum()
    output = []
    for idx in df.index:
        output.append({"count":df.ix[idx]["count"], "date":idx.strftime('%Y-%m-%d')})
    return Response(json.dumps(output), mimetype='application/json')

@app.route('/api/arima', methods=['POST'])
def arima():
    # read post
    indicator = request.json["data"]
    duration = request.json["duration"]
    p = request.json["p"]
    d = request.json["d"]
    q = request.json["q"]
    # create df from json
    counts = []
    dates = []
    for ind in indicator:
        counts.append(ind['count'])
        dates.append(pd.to_datetime(str(ind['date'])))
    indicatordf = pd.Series(counts, index=dates)
    # get model
    res = model.ARIMA_fc(indicatordf, duration, p, d, q)
    # transform to output
    out = apiout(res,indicatordf,int(duration)) 
    return Response(json.dumps(out), mimetype='application/json')

# random forest
@app.route('/api/rf', methods=['POST'])
def rfpost():
    # read post
    indicator = request.json["data"]
    duration = request.json["duration"]
    p = request.json["p"]
    # create df from json
    counts = []
    dates = []
    for ind in indicator:
        counts.append(ind['count'])
        dates.append(pd.to_datetime(str(ind['date'])))
    indicatordf = pd.Series(counts, index=dates)
    # get model
    feauters = []
    labels = []
    for i,real in enumerate(indicatordf):
        if i>=len(indicatordf)-1-int(duration) and i>=int(p):
            ftuple = []
            for f in indicatordf[i-int(p):i]:
                ftuple.append(f)
            feauters.append(ftuple)
            labels.append(float(real))
    res = model.RF_fc(feauters, labels)   
    # transform to output
    out = apiout(res,indicatordf,int(duration))
    return Response(json.dumps(out), mimetype='application/json')

# normalize event registry counts
@app.route('/api/ernorm', methods=['POST'])
def ernorm():
    ers = request.json["data"]
    erdfs = []
    for er in ers:
        counts = []
        dates = []
        for e in er:
            count = e['count']
            date = pd.to_datetime(str(e['date']))
            counts.append(count)
            dates.append(date)
        erdf = pd.Series(counts, index=dates)
        erdf_norm = (erdf - erdf.min()) / (erdf.max() - erdf.min())
        erdfs.append(erdf_norm)
    out = apioutseries(erdfs)
    for er in erdfs:
        f = {"win":4, "max":True, "min":True, "avg":True}
        extract_f(er, f)
    return Response(json.dumps(out), mimetype='application/json')

# nowcasting call
# receives json object with er counts ("er") and indicator count ("indicator")
@app.route('/api/nc', methods=['POST'])
def nc():
    # get data from post
    d = int(request.json["d"])
    ers = request.json["er"]
    indicator = request.json["indicator"]
    model_name = request.json["model"]

    #extract features from er and indicator
    ext = extract(ers, indicator, d)
    
    # final holders
    features = []
    labels = []

    # input dataframes
    erdfs = ext[0]
    indf = ext[1]
    lbldf = ext[2]
 
    # record output dates
    dateout = []

    # construct (merge) features and labels
    for i,idx in enumerate(indf.index):
        # only generate features for data present in all er dataframes and indicator dataframe
        all_have = True

        # temporary feature row tuple
        f_tuple = []

        # check if indicator has data for the current date
        if (any(indf.index == idx) == False or any(lbldf.index == idx) == False):
            all_have = False      
 
        # get er data 
        for erdf in erdfs:
            if any(erdf.index == idx):
                f_tuple.extend(erdf.ix[idx])
            else:
                all_have = False
        
        # append tuple to the feature set
        if all_have == True:
            f_tuple.extend(indf.ix[idx])
            label = lbldf.ix[idx]
            features.append(f_tuple)
            labels.append(label)
            dateout.append(pd.to_datetime(idx).strftime("%Y-%m-%d"))

    # model
    if model_name == "RF":
        res = model.RF_fc(features, labels)
    elif model_name == "SVM":
        res = model.SVM_fc(features, labels)
    elif model_name == "MNB":
        clf = model.newMNB()
        res = model.model_fc(clf, features, labels)
    elif model_name == "SVR":
        clf = model.newSVR()
        res = model.model_fc(clf, features, labels)
    else:
        res = model.RF_fc(features, labels)

    return Response(json.dumps(apincout(res, dateout)), mimetype='application/json')

# extract features from one or more er dataframes and one indicator dataframe
# receives json objects
def extract(ers, indicator, d):
    # er counts to df
    ers = request.json["er"]
    erdfs = []
    for er in ers:
        counts = []
        dates = []
        for e in er:
            count = e['count']
            date = pd.to_datetime(str(e['date']))
            counts.append(count)
            dates.append(date)
        erdf = pd.Series(counts, index=dates)
        erdfs.append(erdf)
    
    # indicator to df
    counts = []
    dates = []
    for ind in indicator:
        counts.append(ind['count'])
        dates.append(pd.to_datetime(str(ind['date'])))
    indicatordf = pd.Series(counts, index=dates)
   
    # features from er
    erfdfs = []
    for erdf in erdfs:
        f = {"win":d, "max":True, "min":True, "avg":True}
        erfdf = extract_f(erdf, f)
        erfdfs.append(erfdf)
   
    # features from indicator
    f = {"win":d, "max":True, "min":True, "avg":True}
    er_f = extract_f1(indicatordf, f)
    indif = er_f[0]
    lbl = er_f[1]
    
    return erfdfs, indif, lbl

# extract feauters from dataframe - the count for the current date is a lable
# return tuple (features, label)
def extract_f1(df, fs):
    feauters = []
    labels = []
    dates = []
    win = []
    for i,idx in enumerate(df.index):
        if i >= fs["win"]:
            f = list(win)
            if fs["max"] == True:
               f.append(max(win))
            if fs["min"] == True:
               f.append(min(win))
            if fs["avg"] == True:
                f.append(avg(win))
            c = df.ix[idx]
            # bake in
            feauters.append(f)
            labels.append(c)
            dates.append(idx.strftime('%Y-%m-%d'))
            # next step
            win.pop(0)
            win.append(df.ix[idx])
        else:
            win.append(df.ix[idx])

    dfout = pd.Series(feauters, index=dates)
    dfout_lbl = pd.Series(labels, index=dates)
    return dfout, dfout_lbl

def extract_f(df, fs):
    feauters = []
    labels = []
    dates = []
    win = []
    for i,idx in enumerate(df.index):
        if i >= fs["win"]-1:
            win.append(df.ix[idx])
            f = list(win)
            if fs["max"] == True:
               f.append(max(win))
            if fs["min"] == True:
               f.append(min(win))
            if fs["avg"] == True:
                f.append(avg(win))

            c = df.ix[idx]
            # bake in
            feauters.append(f)
            labels.append(c)
            dates.append(idx.strftime('%Y-%m-%d'))
            # next step
            win.pop(0)
        else:
            win.append(df.ix[idx])

    dfout = pd.Series(feauters, index=dates)
    return dfout

def avg(li):
    return sum(li)/len(li)

###################
# Internal
###################

# Main
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=8181)
