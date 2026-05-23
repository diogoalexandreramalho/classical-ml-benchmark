import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import naive_bayes as nb
import KNN as knn
import Decision_Tree as dt

def standardScaler(trnX, tstX, trnY, tstY):

    scaler = StandardScaler()
    trnX = scaler.fit_transform(trnX) 
    tstX = scaler.transform(tstX)
    
    return trnX, tstX, trnY, tstY


def minMaxScaler(trnX, tstX, trnY, tstY):

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX) 
    tstX = scaler.transform(tstX)
    
    return trnX, tstX, trnY, tstY
