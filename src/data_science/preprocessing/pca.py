import numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA


#This PCA does not balance or normalize data
def pca(dataset, name):
    data = dataset.copy()

    if(name == "PD"):
        clss = data.pop('class')
        
        pca = PCA(n_components=100, svd_solver='full')
        pca.fit_transform(data)
        
        #Create new_data and add class
        new_data = pd.DataFrame(pca.components_, columns=data.columns)
        new_data['class'] = clss

        return new_data

#Testing        
#dataset = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
#new_data = pca(dataset, "PD")


