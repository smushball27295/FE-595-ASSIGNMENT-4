#Question 1

def lin():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    boston_dataset = load_boston()

    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston.head()
    boston['MEDV'] = boston_dataset.target


    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
    Y = boston['MEDV']    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    

    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    lin_model.intercept_

    print(max(abs(lin_model.coef_)))


lin()    

####question 2 
def kmeans():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    from sklearn import datasets
    import pandas as pd
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
    lin()
  



#We can see that the optimal number of clusters is 3

















