# -*- coding: utf-8 -*-
"""
@author: jason
"""

from pandas import Series, DataFrame, concat
from numpy import nan, isnan, array, append
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from pickle import dump, load, HIGHEST_PROTOCOL

def CheckSeriesValidity(series):
    validway = {1:1, 2:3, 3:1, 4:6, 5:3, 6:1}
    nanway = {1:2, 2:4, 3:5, 4:7, 5:7, 6:7}
    if isinstance(series, Series):
        series[series<0] = nan
        label = 1
        for index, value in series.iteritems():
            if isnan(value):
                label = nanway[label]
                if label == 7:
                    return False
            else:
                label = validway[label]
        return True
    else:
        raise TypeError("Input expected type pandas.Series, received type", type(series))

def Interpolate(data):
    if isinstance(data, Series):
        return data.interpolate(limit_direction="both")
    elif isinstance(data, DataFrame):
        return data.interpolate(limit_direction="both")
    else:
        raise TypeError("Input expected type pandas.Series, received type", type(series))

class EnsembleModel:
    def __init__(self, structure={"A":"XGB", "B":"MLP", "C":"XGB"}, name="Defalt"):
        self.structure = structure
        self.models = dict()
        self.name = name
        self.__create()

    def __create(self):
        for k, v in self.structure.items():
            if v == "XGB":
                self.models[k] = GradientBoostingRegressor()
            elif v == "MLP":
                self.models[k] = MLPRegressor()
            elif v == "LR":
                self.models[k] = LinearRegression()
            else:
                raise ValueError("Invalid structure '%s' to create."%(self.structure)) 
        
    def fit(self, x, y, features):
        if isinstance(features, dict):
            self.models["A"].fit(x[features["A"]], y)
            self.models["B"].fit(x[features["B"]], y)
            
            A_pred = self.models["A"].predict(x[features["A"]])
            #print("RMSE: %f"%(round(sqrt(mean_squared_error(A_pred, y)), 6)))
            B_pred = self.models["B"].predict(x[features["B"]])
            #print("RMSE: %f"%(round(sqrt(mean_squared_error(B_pred, y)), 6)))
            
            preds = DataFrame({"A_pred":A_pred, "B_pred":B_pred})
            C = x[features["C"]].reset_index(drop=True)
            C = concat([preds, C], axis=1)
            self.models["C"].fit(C, y)
            print(self.name, C.shape)
            C_pred = self.models["C"].predict(C)
            self.rmse = round(sqrt(mean_squared_error(C_pred, y)), 6)
            #print("RMSE: %f"%(self.rmse))
            
        else:
            raise TypeError("Input expected type dict, received type", type(features))
    
    def predict(self, x, features):
        if isinstance(x, Series):
            if isinstance(features, dict): 
                A = x[features["A"]].values.reshape(1, -1)
                A_pred = self.models["A"].predict(A)
                B = x[features["B"]].values.reshape(1, -1)
                B_pred = self.models["B"].predict(B)
                C = x[features["C"]].values.reshape(1, -1)
                C = append(append(A_pred, B_pred), C).reshape(1, -1)
                return self.models["C"].predict(C)
            else:
                raise TypeError("Input expected type dict, received type", type(features))
        else:
            raise TypeError("Input expected type pandas.Series, received type", type(x))
    
    def save(self, filename):
        if isinstance(filename, str):
            with open(filename, 'wb') as handle:
                dump(self, handle, protocol=HIGHEST_PROTOCOL)
        else:
            raise TypeError("Input expected type string, received type", type(filename))
    
    def load(filename):
        if isinstance(filename, str):
            with open(filename, 'rb') as handle:
                return load(handle)
        else:
            raise TypeError("Input expected type string, received type", type(filename))
