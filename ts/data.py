import sys
from tinydb import TinyDB, where
from Quandl import Quandl
import pandas as pd
import datetime
import urllib
import urllib2
import json
import re, sys
from eventregistry import *

### DATABASE

## connect with database file
dbIndicators = TinyDB('data/dbIndicators.json')

## Define tables

# table with indicatos values: name(string) - ABCD/Efghijkl, date(tripe) - (year, month, day); value (float)
tblIndicators = dbIndicators.table("indicators")

# table with list of indicators in the store: name(string) - ABC/Defghijkl; resolution(string) - month; count(int) - 120  
tblIndex = dbIndicators.table("index")

### DB Manipulation

def clean():
    tblIndicators.purge()
    tblIndex.purge()

# check if there is a timeseries in the database already
def isStored(name):
    res = tblIndex.get(where('name') == name)
    if res == None:
        return False
    else:
        return True

def listIndicators():
    return tblIndex.all()

def printIndicators():
    indicators = listIndicators()
    for i in indicators:
        print i["nae"], i["desc"]

# get indicator from database
def getIndicatorFromDb(indicator_name):
    res = tblIndicators.search(where('name') == indicator_name)
    return res

### Quandl queries

# search for an indicator
def searchQuandl(term):
    res = Quandl.search(query = term, page = 1)
    return res

# get indicator data from Quandl and store it into the database
def getIndicatorFromQuandl(indicator_name):
    resolution = 'monthly'
    print "Getting "+indicator_name+" indicator from Quandl"
    today = str(datetime.datetime.now().year)+"-"+str(datetime.datetime.now().month)+"-"+str(datetime.datetime.now().day)
    data = Quandl.get(indicator_name, authtoken='5UND8fCsD7oWNEs1fpfa', trim_start='1900-01-01', trim_end=today, collapse=resolution, transformation='none', returns='numpy')
    datapoints = []
    for datapoint in data:
        dt = datapoint[0]
        datapoints.append({"name":indicator_name, "date":(dt.year, dt.month, dt.day), "value": datapoint[1] })
    # insert multiple records to the table
    tblIndicators.insert_multiple(datapoints)
    # get name i.e. description of the indicator
    meta = searchQuandl(indicator_name);
    desc = meta[0]["name"]
    # insert new record into indicators index
    tblIndex.insert({"desc": desc, "name": indicator_name, "resolution": resolution, "count": len(datapoints)})
    print "Sucessful insert of "+str(len(datapoints))+" records."+indicator_name
    return datapoints

def toValArray(data):
    arr = []
    for dp in data:
        arr.append(dp["value"])
    return arr

# event registry correlate

def erGetCountsConcept(input_concepts):
    er = EventRegistry()
    er.login("mario.karlovcec@gmail.com", "jerneja08")
    q = GetCounts(input_concepts)
    ret = er.execQuery(q)
    return ret

def erCorrelateConcept(input_concept):
    er = EventRegistry()
    er.login("mario.karlovcec@gmail.com", "jerneja08")
    corr = GetTopCorrelations(er)
    counts = GetCounts(er.getConceptUri(input_concept))
    corr.loadInputDataWithCounts(counts)
    conceptInfo = corr.getTopConceptCorrelations(conceptType = ["person", "loc", "org", "wiki"], exactCount = 5, approxCount = 5)
    return conceptInfo

def erCorrelateDf(indicatordf):
    er = EventRegistry()
    er.login("mario.karlovcec@gmail.com", "jerneja08")
    corr = GetTopCorrelations(er)
    arr = []
    
    x2 = pd.date_range(indicatordf.index[0],'2016-03-01',freq='1D')
    df2 = indicatordf.reindex(x2)
    df2 = df2.interpolate(method='linear')
    df2 = df2.ix['2013-01-01':'2016-03-01']
   
    for i,j in zip(df2, df2.index):
        arr.append((str(j.date()), i))
    
    corr.setCustomInputData(arr)
    conceptInfo = corr.getTopConceptCorrelations(conceptType = ["person", "loc", "org", "wiki"], exactCount = 5, approxCount = 5)
    correlated_concepts = []

    for c in conceptInfo["news-concept"]["approximateCorrelations"][0:10]:
        correlated_concepts.append(c["conceptInfo"]["uri"])

    for c in conceptInfo["news-concept"]["exactCorrelations"][0:10]:
        correlated_concepts.append(c["conceptInfo"]["uri"])

    counts = erGetCountsConcept(correlated_concepts)
    
    corrs = []
    for k in counts.keys():
        vals = []
        dates = []
        for p in counts[k]:
            vals.append(p['count'])
            dates.append(pd.to_datetime(p['date']))
        corrs.append(pd.Series(vals, index=dates))

    return corrs,correlated_concepts

### MAIN

# get indicator values as list of floats
def getIndicator(indicator_name):
    if (isStored(indicator_name) == False):
        indicator = getIndicatorFromQuandl(indicator_name)
    else:
        indicator = getIndicatorFromDb(indicator_name)
    return toValArray(indicator)

# get indicator values as dataframe with datetime as index
def getIndicatorDf(indicator_name):
    if (isStored(indicator_name) == False):
        indicator = getIndicatorFromQuandl(indicator_name)
    else:
        indicator = getIndicatorFromDb(indicator_name)
    vals = []
    dates = []
    for p in indicator:
        vals.append(p['value'])
        dates.append(pd.to_datetime(str(p['date'][0])+"-"+str(p['date'][1])+"-"+str(p['date'][2])))
        #dates.append(datetime.date(p['date'][0], p['date'][1], p['date'][2]))
        #dates.append(pd.tslib.TimestampTimestamp(p['date'][0]+'-'+p['date'][1]+'-'+p['date'][2], tz=None))
    return pd.Series(vals, index=dates)

def getIndicatorFull(indicator_name):
    if (isStored(indicator_name) == False):
        indicator = getIndicatorFromQuandl(indicator_name)
    else:
        indicator = getIndicatorFromDb(indicator_name)
    return indicator

def getTestData():
    return [10,8,12,10,8,12,10,8,12,10,8,12,10,8,12,10]
