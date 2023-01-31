import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

Data = pd.read_csv('DataPart03.csv')

X = Data.drop('Sell Price($)', axis=1)
X=pd.get_dummies(X)
print(X)
X = np.array(X)

Y = Data['Sell Price($)']
Y = np.array(Y)

ModelPredict = LinearRegression().fit(X, Y)


Output = 'OutputP.sav'
pickle.dump(ModelPredict, open(Output, 'wb'))
 
LoadModel = pickle.load(open(Output, 'rb'))
Result = LoadModel.predict([[55000,5,1,0,0]])
print(Result)

# Pickle
# Good for small models with few parameters.
# Allows saving model in very little time.