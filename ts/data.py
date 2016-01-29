import sys
from tinydb import TinyDB, where
from Quandl import Quandl
import datetime

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
        print i["name"], i["desc"]

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

### MAIN

# get indicator values as list of floats
def getIndicator(indicator_name):
    if (isStored(indicator_name) == False):
        indicator = getIndicatorFromQuandl(indicator_name)
    else:
        indicator = getIndicatorFromDb(indicator_name)
    return toValArray(indicator)

