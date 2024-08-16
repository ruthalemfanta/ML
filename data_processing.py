# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading from a data set

dataset = pd.read_csv('/Users/rutha./Downloads/1. Welcome to the course!/Machine Learning A-Z New/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#filling missing data 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


#catagorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categories=[0])
x = onehotencoder.fit_transform(x).toarray()
