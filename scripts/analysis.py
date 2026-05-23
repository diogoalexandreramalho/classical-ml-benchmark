import pandas as pd
import numpy as np

import data_balancing as balance
import Normalize as norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import KNN as knn
import Decision_Tree as dt
import naive_bayes as nb
import RandomForest as rf
import GradientBoost as gb
import XGBoost as xgb

import print_statistics as stats

import time
import datetime


def classification(data, source, analysis):

    if source == "CT":
        # get 1000 samples per class and get new data set
        data = data.groupby('Cover_Type').apply(lambda s: s.sample(1000))
        target = 'Cover_Type'
    else:
        target = 'class'

    balanced = True
    normalized = True 

    
    # separates the dataset in training and testing sets
    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    # normalize and balance the dataset
    trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
    trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)

    if source == "PD":
        start = time.time()
        # find best model for each classifier
        #print("NB")
        #nb_report = nb.naive_bayes(trnX, tstX, trnY, tstY, labels, False)
        #print("KNN")
        #knn_report = knn.k_near_ngb(trnX, tstX, trnY, tstY, labels, True)
        #print("DT")
        #dt_report = dt.decision_tree(trnX, tstX, trnY, tstY, labels, True, False)
        #print("RF")
        #rf_report = rf.random_forest(trnX, tstX, trnY, tstY, labels, True)
        #print("GB")
        #gb_report = gb.gradient_boost(trnX, tstX, trnY, tstY, labels, True)
        print("XGB")
        xgb_report = xgb.xg_boost(trnX, tstX, trnY, tstY, labels, True)
        end = time.time() - start
        time1 = str(datetime.timedelta(seconds=end)) 
        print("Time: " + time1)
    else:
        # find best model for each classifier
        start = time.time()
        #print("NB")
        #nb_report = nb.naive_bayes_CT(trnX, tstX, trnY, tstY, labels)
        #print("KNN")
        #knn_report = knn.k_near_ngb_CT(trnX, tstX, trnY, tstY, labels, True)
        #print("DT")
        #dt_report = dt.decision_tree_CT(trnX, tstX, trnY, tstY, labels, True, False)
        #print("RF")
        #rf_report = rf.random_forest_CT(trnX, tstX, trnY, tstY, labels, True)
        #print("GB")
        #gb_report = gb.gradient_boost_CT(trnX, tstX, trnY, tstY, labels, True)
        print("XGB")
        xgb_report = xgb.xg_boost_CT(trnX, tstX, trnY, tstY, labels, True)
        end = time.time() - start
        time1 = str(datetime.timedelta(seconds=end)) 
        print("Time: " + time1)

    reports = [xgb_report]
    #reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]
    
    if source == "PD":
        stats.print_analysis(reports, (balanced, normalized))
    else:
        stats.print_analysis_CT(reports, (balanced, normalized))



def produce_analysis():

    #data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
    data = pd.read_csv('Data/covtype.csv', sep=',', decimal='.')

    classification(data, "CT", True)


produce_analysis()