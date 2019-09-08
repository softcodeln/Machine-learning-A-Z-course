# Polynomial Regression

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

#Fitting Linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualinsing the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('level')
plt.ylabel('Salary')

plt.show()
#Visualinsing the Polynomial Regression
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

# Predicting new result with linear Regression
lin_reg.predict([[6.5]])

#Predicting new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))