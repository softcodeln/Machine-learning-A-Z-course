# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:13:25 2019

@author: Lucky
"""
# Hierarchical Clustering

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
datasets = pd.read_csv('Mall_Customers.csv')
X = datasets.iloc[:, [3, 4]].values

# Using the dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting  hierarchical clustering to Mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising Clusters
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1] ,s=100, c='red', label='Carefull' )
plt.scatter(X[y_hc==1, 0],X[y_hc==1, 1] , s=100, c='blue', label='Standard' )
plt.scatter(X[y_hc==2, 0],X[y_hc==2, 1] , s=100, c='green', label='Target')
plt.scatter(X[y_hc==3, 0],X[y_hc==3, 1] , s=100, c='cyan', label='Careless')
plt.scatter(X[y_hc==4, 0],X[y_hc==4, 1] , s=100, c='magenta', label='Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()  