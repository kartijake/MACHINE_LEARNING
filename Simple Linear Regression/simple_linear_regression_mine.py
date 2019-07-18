# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:40:43 2019

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa
#dataset
dataset= pa.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values #qmatrix of independent variable
Y=dataset.iloc[:,1].values
"""
#find mean 
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:])
X[:,1:]=imputer.transform(X[:,1:3])
#encoding catogorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0]);
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)"""
#spliting into test and traing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#fitting simple linear regression on the training set 
#linear model library
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
# predicter
y_pred=regressor.predict(X_test)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience (Traing set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualize test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience (Traing set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()