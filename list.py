from ts import data
indicators =  data.listIndicators()
for i in indicators:
    print(i["name"], i["desc"])

