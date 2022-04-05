#%%
# Importing the libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# %%
regressor = LinearRegression()
# %%
from numpy import *  
import matplotlib.pyplot as plt  
  
  
def loadDataSet(fileName):   
    numFeat = len(open(fileName).readline().split('\t')) - 1       #得到特征值的个数  
    dataMat = []; labelMat = []  
    fr = open(fileName)                     #打开文件  
    for line in fr.readlines():               #读取整行  
        lineArr =[]  
        curLine = line.strip().split('\t')           #将一行的不同特征分开  
        for i in range(numFeat):  
            lineArr.append(float(curLine[i]))  
        dataMat.append(lineArr)  
        labelMat.append(float(curLine[-1]))  
    return dataMat,labelMat  
  
  
def standRegres(xArr,yArr):  
    xMat = mat(xArr); yMat = mat(yArr).T  
    xTx = xMat.T*xMat  
    ws = xTx.I * (xMat.T*yMat)       #求 w=(x.T*x).I*x.T*y  
    return ws  
  
a,b=loadDataSet('ex0.txt')  
ws=standRegres(a,b)  
print ws  
x=arange(0,1,0.01)  
plt.plot([i[1] for i in a],b,'or')  
plt.plot(x,float(ws[0])+float(ws[1])*x,'g')  
plt.show()  

#%%
# Simple Linear Regression

# Importing the dataset

X_data = pd.read_csv('D:\\北美程序员之路\\Leetcode\\kaggle\\linear regression\\train_examples.csv')
y_data = pd.read_csv('D:\\北美程序员之路\\Leetcode\\kaggle\\linear regression\\train_labels.csv')
#%%
X = X_data.iloc[:,2:]
y = y_data.iloc[:,1:]
#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#%%
mean_absolute_error(y_test,y_pred)
# 只取一个特征的时候得到的mae是675
# 取11个特征的时候得到的mae是485
# 由此可见,随着特征数增多,其mae也在增多
# %%
