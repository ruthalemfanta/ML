# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading from a data set

dataset = pd.read_csv('/Users/rutha./Downloads/1. Welcome to the course!/Machine Learning A-Z New/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values #dependent

#filling missing data 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


#catagorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder()
x_country = onehotencoder.fit_transform(x[:, [0]]).toarray()
x = np.concatenate((x_country, x[:, 1:]), axis=1)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#training set and test test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#Ml models are based of euclidean  
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
