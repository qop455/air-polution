# -*- coding: utf-8 -*-
"""
@author: jason
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import mopy as mo
from sklearn.model_selection import train_test_split
#plt.rcParams['figure.figsize'] = [16.0, 8.0]
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"
def main(station_id):
    t = time.time()
    
    station = station_id
    data = pd.read_csv("../data/central_reg/filted/" + str(station) + ".csv")
    f=12
    n=6
    
    for i in range(f):
        data["PM25_%d"%(i+1)] = data["PM25"].shift(i+1)
    for i in range(n):
        data["F_TEMP_%d"%(i+1)] = data["TEMP"].shift(-(i+1))
        data["F_RH_%d"%(i+1)] = data["RH"].shift(-(i+1))
        data["F_RAIN_%d"%(i+1)] = data["RAIN"].shift(-(i+1))
        data["F_WD_%d"%(i+1)] = data["WD"].shift(-(i+1))
        data["F_WS_%d"%(i+1)] = data["WS"].shift(-(i+1))
    for i in range(n):
        data["Target_%d"%(i+1)] = data["PM25"].shift(-(i+1))
    
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
    features = dict()
    for i in range(n):
        if i == 0:
            col_F["F%d"%(i+1)] = [f for f in data.columns.tolist() if "F_" in f and "_%d"%(i+1) in f]
        else:
            col_F["F%d"%(i+1)] = col_F["F%d"%(i)] + [f for f in data.columns.tolist() if "F_" in f and "_%d"%(i+1) in f]
    
        features["F%d"%(i+1)] = {"A":col_PM25, "B":col_PM25, "C":col_F["F%d"%(i+1)]}
    
    droplist = list()
    for index, row in data.iterrows():
        if mo.CheckSeriesValidity(row[col_PM25]) and mo.CheckSeriesValidity(row[col_TEMP]) and mo.CheckSeriesValidity(row[col_RH]) and mo.CheckSeriesValidity(row[col_RAIN]) and mo.CheckSeriesValidity(row[col_WD]) and mo.CheckSeriesValidity(row[col_WS]):
            pass
        else:
            droplist.append(index)
    
    data.drop(data.index[[droplist]], inplace=True)
    data = mo.Interpolate(data)
    #train, test = train_test_split(data, test_size = 0.01) 
    train, test = data[:-740], data[-740:]
    print("Train shape:%s Test shape:%s"%(train.shape,test.shape)) 
    ensembleStructure = {"A":"XGB", "B":"MLP", "C":"XGB"}
    
    models = dict()
    for i in range(n):
        
        models["F%d"%(i+1)] = mo.EnsembleModel(ensembleStructure, "F%d"%(i+1))
        models["F%d"%(i+1)].fit(train, train["Target_%d"%(i+1)], features["F%d"%(i+1)])
    
    print("Fit time: %d s."%(time.time() - t))
    
    for i in range(n):
        mo.save_ensemble(models["F%d"%(i+1)], "../output/model/" + str(station) + "/F%d"%(i+1))
        
    models = None
    models = dict()
    
    for i in range(n):
        path = "../output/model/" + str(station) + "/F%d"%(i+1)
        models["F%d"%(i+1)] = mo.load_ensemble(path)
    
    df_columns = ("datetime", "station", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5", "pred_6")
    predict = pd.DataFrame(columns=df_columns)
    
    for index, row in test.iterrows():
        preds = [row['datetime']]
        preds.extend([station])
    
        for i in range(n):
            model = models["F%d"%(i+1)]
            pred = model.predict(row, features["F%d"%(i+1)])
            pred = [ '%.2f' % abs(float(elem)) for elem in pred]
            preds.extend(pred)
    
        preds = pd.DataFrame([preds], columns=df_columns)
        predict = predict.append(preds)
    
        '''
        plt.title('6 hours prediction')
        plt.xlabel('Time Line')
        plt.ylabel('PM2.5 Value')
        real = row[col_PM25 + col_Target].tolist()
        predict = row[col_PM25].tolist() + preds.tolist()
        plt.plot(range(n+f+1), predict, 'r--', label="Predict")
        plt.plot(range(n+f+1), real, 'b-', label="Real")
        plt.xticks(range(n+f+1), [str(x-12) for x in range(n+f+1)])
        plt.xticks(rotation=90)
        plt.ylim([min(min(predict), min(real)) - 5, max(max(predict), max(real)) + 5])
        plt.axhline(y=15.5, color='g', linestyle='--', xmin=0.01, xmax=0.99, alpha=0.8)
        plt.axhline(y=35.4, color='y', linestyle='--', xmin=0.01, xmax=0.99, alpha=0.8)
        plt.axhline(y=54.5, color='r', linestyle='--', xmin=0.01, xmax=0.99, alpha=0.8)
        plt.axvline(x=12, color='r', ymin=0.01, ymax=0.99, alpha=0.8)
        plt.legend()
        plt.savefig("../output/picture/"+str(index)+".png")
        plt.clf()
        '''
    
    predict.to_csv("predict_%d.csv"%(station), index=False)

if __name__ == "__main__":
    for i in range(11):
        print(i)
        main(i+1)