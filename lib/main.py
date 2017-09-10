# -*- coding: utf-8 -*-
"""
@author: jason
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import mopy as mo

TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

data = pd.read_csv("../data/central_reg/filted/1.csv")
data = data[:1000]

for i in range(12):
    data["PM25_"+str(i+1)] = data["PM25"].shift(i+1)
for i in range(6):
    data["F_TEMP_"+str(i+1)] = data["TEMP"].shift(-(i+1))
    data["F_RH_"+str(i+1)] = data["RH"].shift(-(i+1))
    data["F_RAIN_"+str(i+1)] = data["RAIN"].shift(-(i+1))
    data["F_WD_"+str(i+1)] = data["WD"].shift(-(i+1))
    data["F_WS_"+str(i+1)] = data["WS"].shift(-(i+1))
for i in range(6):
    data["Target_"+str(i+1)] = data["PM25"].shift(-(i+1))

timeFeature = pd.DataFrame(index=["hour", "week", "month"])
for index, item in data.datetime.iteritems():
    time = datetime.strptime(item, TIMEFORMAT)
    data.set_value(index, 'hour', time.hour)
    data.set_value(index, 'week', time.weekday())
    data.set_value(index, 'month', time.month)

time_dummies = pd.get_dummies(data["hour"]).rename(columns = lambda x:"h_" + str(int(x)))
week_dummies = pd.get_dummies(data["week"]).rename(columns = lambda x: "w_" + str(int(x+1)))
month_dummies = pd.get_dummies(data["month"]).rename(columns = lambda x: "m_" + str(int(x)))

data = pd.concat([data, time_dummies, week_dummies, month_dummies], axis=1)
data.drop(["hour", "week", "month"], inplace=True, axis=1)

col_PM25 = [f for f in data.columns.tolist() if "PM25" in f]
col_TEMP = [f for f in data.columns.tolist() if "TEMP" in f]
col_RH = [f for f in data.columns.tolist() if "RH" in f]
col_RAIN = [f for f in data.columns.tolist() if "RAIN" in f]
col_WD = [f for f in data.columns.tolist() if "WD" in f]
col_WS = [f for f in data.columns.tolist() if "WS" in f]

for index, raw in data.iterrows():
    if mo.CheckSeriesValidity(raw[col_PM25]):
        pass
    else:
        data.drop(data.index[[index]], inplace=True)
        