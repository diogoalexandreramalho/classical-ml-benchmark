import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_functions as charts
import Normalize as norm
from itertools import cycle, islice
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import SelectKBest, f_classif
from yellowbrick.cluster import KElbowVisualizer
import mpl_toolkits.mplot3d.axes3d as p3



def run(source,data):

    dataPD = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)    
    dataCT = pd.read_csv('Data/covtype_test.data', sep=',', decimal='.')    



    if source == "PD":
        target = "class"
        data = dataPD
    else:
        data = dataCT
        target = "Cover_Type"

    data_norm = norm.normalization(source,data)
    y = data_norm.pop(target).values
    X = data_norm.values

    visualizer = KElbowVisualizer(cluster.KMeans(), k=(1,12))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    



    #print("Best k for Clusters :" + str(k))
    #print("Sum of squared distances :" + str(best_kmeans_inertia))

    ### Silhouette and Rand index Scores for the best K
    #print("Silhouette:",metrics.silhouette_score(X, y_pred))
    #print("RI[KMeans] =",adjusted_rand_score(y, y_pred))


def statistics(source,data):

    data_norm = norm.normalization(source,data)
   
    if source == "PD":
        target = "class"
        number_of_clusters = 4
    else:
        target = "Cover_Type"
        number_of_clusters = 3

    y = data_norm.pop(target).values
    X = data_norm.values

    for i in range(len(y)) :
        y[i] = int(y[i])

    kmeans_model = cluster.KMeans(n_clusters=number_of_clusters, random_state=1).fit(X)
    y_pred = kmeans_model.labels_

    print(y_pred)

    selector = SelectKBest(f_classif, k=3)
    kb = selector.fit_transform(X,y)
    X = pd.DataFrame(kb).values

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(y_pred):
        ax.scatter(X[y_pred == l, 0], X[y_pred == l, 1], X[y_pred == l, 2],
                color=plt.cm.jet(float(l) / np.max(y_pred + 1)), s=20, edgecolor='k')
    plt.title('Clustering Solution')
    plt.show()




    print("2.Clustering :")
    print("a) Number of Clusters : " + str(number_of_clusters))
    print("b) Sum of squared distances :", kmeans_model.inertia_)
    print("c) Average Silhouette Coeficent :",metrics.silhouette_score(X, y_pred))
    print("d) Rand Index : ",adjusted_rand_score(y, y_pred))


