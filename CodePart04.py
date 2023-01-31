import numpy as np
import pandas as pd
import csv

Data = pd.read_csv('DataPart04.csv')

X = Data['Car Model']
X = pd.get_dummies(X)
Y = Data.drop('Car Model', axis=1)

DataHeadX = list(X[1:1])
DataHeadY = list(Y[1:1])
DataHead = DataHeadX+DataHeadY

DataX = np.asarray(X)
DataY = np.asarray(Y)
Data = np.concatenate((DataX, DataY), axis=1)

pd.DataFrame(Data).to_csv('DataPart04.csv', index_label = "Index", header  = DataHead) 

