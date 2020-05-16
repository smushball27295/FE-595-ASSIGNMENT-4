# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:43:18 2020

@author: ramesh
"""

####question 2 
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn import datasets
import pandas as pd
    
def kmeans():
    iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

    x = iris_df.iloc[:, [0,1,2,3]].values


    kmeans5 = KMeans(n_clusters=5)
    y_kmeans5 = kmeans5.fit_predict(x)
    print(y_kmeans5)
    
    
    Error =[]
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i).fit(x)
        kmeans.fit(x)
        Error.append(kmeans.inertia_)
    import matplotlib.pyplot as plt
    plt.plot(range(1, 11), Error)
    plt.title('Elbow method')
    plt.xlabel('No of clusters')
    plt.ylabel('Error')
    plt.show()
        
if __name__=="__main__":
    kmeans()
  
#We can see that the optimal number of clusters is 3

