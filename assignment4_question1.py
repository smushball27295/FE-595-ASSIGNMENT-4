#Question 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def lin():
    
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


if _name_=="_main_":
    lin()    

















