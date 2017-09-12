# -*- coding: utf-8 -*-
"""
@author: jason
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.plotly as py
import mopy as mo
from sklearn.model_selection import train_test_split
#plt.rcParams['figure.figsize'] = [16.0, 8.0]
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

data = pd.read_csv("../data/central_reg/filted/1.csv")
data = data[:100]

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
'''
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
'''
col_PM25 = [f for f in data.columns.tolist() if "PM25" in f]
col_TEMP = [f for f in data.columns.tolist() if "TEMP" in f]
col_RH = [f for f in data.columns.tolist() if "RH" in f]
col_RAIN = [f for f in data.columns.tolist() if "RAIN" in f]
col_WD = [f for f in data.columns.tolist() if "WD" in f]
col_WS = [f for f in data.columns.tolist() if "WS" in f]
col_Target = [f for f in data.columns.tolist() if "Target" in f]

col_F = dict()
for i in range(6):
    col_F["F"+str(i+1)] = [f for f in data.columns.tolist() if "F_" in f and "_"+str(i+1) in f]

droplist = list()
for index, row in data.iterrows():
    if mo.CheckSeriesValidity(row[col_PM25]) and mo.CheckSeriesValidity(row[col_TEMP]) and mo.CheckSeriesValidity(row[col_RH]) and mo.CheckSeriesValidity(row[col_RAIN]) and mo.CheckSeriesValidity(row[col_WD]) and mo.CheckSeriesValidity(row[col_WS]):
        pass
    else:
        droplist.append(index)
data.drop(data.index[[droplist]], inplace=True)
data = mo.Interpolate(data)
train, test = train_test_split(data, test_size = 0.01) 
ensembleStructure = {"A":"XGB", "B":"MLP", "C":"XGB"}

models = dict()
features = dict()
for i in range(6):
    models["F"+str(i+1)] = mo.EnsembleModel(ensembleStructure)
    features["F"+str(i+1)] = {"A":col_PM25, "B":col_PM25, "C":col_F["F"+str(i+1)]}
    models["F"+str(i+1)].fit(train, train["Target_"+str(i+1)], features["F"+str(i+1)])

for index, row in test.iterrows():
    preds = np.array([])
    for i in range(6):
        pred = models["F"+str(i+1)].predict(row, features["F"+str(i+1)])
        preds = np.append(preds, pred)

    plt.title('6 hours prediction')
    plt.xlabel('Time Line')
    plt.ylabel('PM2.5 Value')
    real = row[col_PM25 + col_Target].tolist()
    predict = row[col_PM25].tolist() + preds.tolist()
    plt.plot(range(19), predict, 'r--', label="Predict")
    plt.plot(range(19), real, 'b-', label="Real")
    plt.xticks(range(19), [str(x+1) for x in range(19)])
    plt.xticks(rotation=90)
    plt.ylim([min(real) - 5, max(real) + 5])
    plt.axhline(y=15.5, color='g', xmin=0.01, xmax=0.99, alpha=0.8)
    plt.axhline(y=35.4, color='y', xmin=0.01, xmax=0.99, alpha=0.8)
    plt.axvline(x=12, color='r', ymin=0.01, ymax=0.99, alpha=0.8)
    plt.legend()
    plt.savefig("../output/picture/"+str(index)+".png")

