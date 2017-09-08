import pandas 
import numpy
import xgboost
import sklearn

def CheckSeriesValidity(series):
    validway = {1:1, 2:3, 3:1, 4:6, 5:3, 6:1}
    nanway = {1:2, 2:4, 3:5, 4:7, 5:7, 6:7}
    if isinstance(series, pandas.Series):
        series[series<0] = numpy.nan
        label = 1
        for index, value in series.iteritems():
            if numpy.isnan(value):
                label = nanway[label]
                if label == 7:
                    return False
            else:
                label = validway[label]
        return True
    else:
        raise TypeError("Input expected type pandas.Series, received type", type(series))

def InterpolateSeries(series):
    if isinstance(series, pandas.Series):
        return series.interpolate(limit_direction="both")
    else:
        raise TypeError("Input expected type pandas.Series, received type", type(series))

class EnsembleModel:
    structure = {"A":"XGB", "B":"MLP", "C":"XGB"}
    models = list()

    def __init__(self, structure):
        self.structure = structure
        self.__create()

    def __create(self):
        for k, v in self.structure.iteritems():
            if v == "XGB":
                self.models.append(xgboost.XGBRegressor())
            elif v == "MLP":
                self.models.append(sklearn.neural_network.MLPRegressor())
            elif v == "LR":
                self.models.append(sklearn.linear_model.LinearRegression())
            else:
                raise ValueError("Invalid structure '%s' to create."%(self.structure)) 
        
    def fit(self, x, y):
        return
    
    def predict(self, x):
        return
    
    
    
    