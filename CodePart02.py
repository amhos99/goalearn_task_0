import numpy as np
import pandas as pd
import math
import csv

Data = pd.read_csv('DataPart02.csv')

X = Data["math"]
X = np.array(X)
Y = Data["cs"]
Y = np.array(Y)

M = B = 0
Iterations = 100000
PointNum = len(X)
LearnRate = 0.0002
CostPrev = 0

Output = open("OutputPart02.csv", "w")

Data = ["Iteration", "M", "B", "Cost"]
csv.writer(Output).writerow(Data)


for i in range(Iterations):
    YPred = B + np.dot(X, M)
    Cost = np.sum(np.dot((Y - YPred).T, Y - YPred))/PointNum
    UB = (np.sum(YPred-Y)*2)/PointNum
    UM = (np.dot((YPred-Y), X)*2)/PointNum
    B = B-LearnRate*UB
    M = M-LearnRate*UM
    if math.isclose(Cost, CostPrev, rel_tol=1e-20):
        break
    CostPrev = Cost
    Data = [i, M, B, Cost]
    csv.writer(Output).writerow(Data)

Output.close()
