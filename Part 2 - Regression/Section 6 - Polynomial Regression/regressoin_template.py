# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 01:56:53 2019

@author: Lucky
"""

#..........................................Regression Template........................................

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2]

# Comparing Salary and level
'''plt.scatter(X,y)
plt.plot(X,y)
plt.title('Salary Vs level')
plt.xlabel('level')
plt.ylabel('Salary')'''

# Encoding the categorical features
'''dummy = pd.get_dummies(dataset.Position)
X = pd.concat([dataset, dummy],axis='columns')
X = X.drop(['Position'],axis='columns')'''

#Avoiding dummy variable trap
'''X = X.drop(['Senior Partner'],axis='columns')'''

#Splitting the data into training and test set
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =   train_test_split(X,y,test_size=0.2,random_state=0)'''

# Fitting Regression Model to the dataset
        #.....................Create your regressor here

# Predicting new result with Regression
y_pred = regressor.predict([[6.5]])

# Visualinsing the Regression results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

# Visualinsing the Regression results with more Resolution or more smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()