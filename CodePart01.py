import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

Data = pd.read_csv('DataPart01.csv')
NewData = Data.dropna()

X = NewData.drop('salary($)', axis=1)
X = np.array(X)

Y = NewData['salary($)']
Y = np.array(Y)

ModelPredict = LinearRegression().fit(X, Y)

Output = open("OutputPart01.txt", "w")
SalaryPredict = ModelPredict.predict([[12, 10, 10]])
Output.write(
    "Experience: 12\tTest_Score: 10\tInterview_Score: 10\tSuggested_Salary($): %d\r\n" % (SalaryPredict))

SalaryPredict = ModelPredict.predict([[2, 9, 6]])
Output.write(
    "Experience: 2\tTest_Score: 9\tInterview_Score: 6\tSuggested_Salary($): %d\r\n" % (SalaryPredict))
Output.close()
