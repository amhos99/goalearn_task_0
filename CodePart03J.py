import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

Data = pd.read_csv('DataPart03.csv')

X = Data.drop('Sell Price($)', axis=1)
X=pd.get_dummies(X)
X = np.array(X)

Y = Data['Sell Price($)']
Y = np.array(Y)

ModelPredict = LinearRegression().fit(X, Y)


Output = 'OutputJ.sav'
joblib.dump(ModelPredict, Output)
 
LoadModel = joblib.load(Output)
Result = LoadModel.predict([[55000,5,1,0,0]])
print(Result)

# joblib
# Ideal for the large models having many parameters.
# Can only save the file to disk and not to a string.