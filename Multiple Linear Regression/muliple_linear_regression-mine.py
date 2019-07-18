# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

dataset=pd.read_csv("50_Startups.csv")

X=dataset.iloc[:,:4].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X=LabelEncoder()
X[:,3]=label_encoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
#avoiding dummy variable trap
X=X[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)