# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding Categorical dataset
dummy = pd.get_dummies(dataset.State)
X=pd.concat([dataset, dummy],axis='columns')
X=X.drop(['State','Profit'],axis='columns')

#Avoiding the dummy variable trap
X = X.drop(['New York'],axis='columns')

'''from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categories='auto')
X = onehotencoder.fit_transform(X).toarray()'''

#Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting the Multi linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting for Test set
y_pred = regressor.predict(X_test)

# Building optimal model by using Backward Elimination
import statsmodels.formula.api as sm

X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis=1 )
X_opt = X[:, [0,1,2,3,4,5]]
regressor_opt = sm.OLS(endog=y, exog=X_opt).fit()
regressor_opt.summary()

X_opt = X[:, [0,1,2,3,5]]
regressor_opt = sm.OLS(endog=y, exog=X_opt).fit()
regressor_opt.summary()


X_opt = X[:, [0,1,2,3]]
regressor_opt = sm.OLS(endog=y, exog=X_opt).fit()
regressor_opt.summary()


X_opt = X[:, [0,1,3]]
regressor_opt = sm.OLS(endog=y, exog=X_opt).fit()
regressor_opt.summary()


X_opt = X[:, [0,1]]
regressor_opt = sm.OLS(endog=y, exog=X_opt).fit()
regressor_opt.summary()

