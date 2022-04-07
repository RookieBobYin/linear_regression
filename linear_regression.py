#%%
# Importing the libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#%%
# Simple Linear Regression

# Importing the dataset
regressor = LinearRegression()
X_data = pd.read_csv('D:\\北美程序员之路\\Leetcode\\kaggle\\linear regression\\train_examples.csv')
y_data = pd.read_csv('D:\\北美程序员之路\\Leetcode\\kaggle\\linear regression\\train_labels.csv')
test_data = pd.read_csv('D:\\北美程序员之路\\Leetcode\\kaggle\\linear regression\\test_examples.csv')
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
mean_absolute_error(y_test,y_pred)
# 只取一个特征的时候得到的mae是675
# 取11个特征的时候得到的mae是485
# 由此可见,随着特征数增多,其mae也在增多
# 现在我们需要去做一个特征选择
#%%
'''
预测数据
'''
x_test = test_data.iloc[:,2:]
y_test = regressor.predict(x_test)
id = pd.DataFrame(test_data.iloc[:,0])
pred = pd.DataFrame(y_test)
pred.columns = ['duration']
res = pd.concat([id,pred],axis = 1)
res =res.set_index('id')
res.to_csv('D:\\北美程序员之路\\Leetcode\\kaggle\\linear regression\\result2022-04-05.csv')
# 结果overfitting了
# 这里的解决方法有1.使用lasso regression去选择feature,这样来提高模型的表现
# 解决方案2.未知
#%%



# %%
# 需要自己实现一个类似功能的linear regression并且得到一个好一点的结果
import numpy as np
from sklearn.linear_model import LinearRegression

#模拟数据
x = np.linspace(0, 10, 50)
noise = np.random.uniform(-1,1,size=50)
y = 5 * x + 6 + noise
#计算损失函数
def compute_cost(x, y, theta):
    y_pred = np.dot(x, theta.T)
    inner = np.power((y_pred-y), 2)
    cost = np.sum(inner, axis=0) / (2 * x.shape[0])
    return cost
#梯度下降
def grandient_descent(x, y, theta, alpha, iters):
    #参数长度
    len_params = theta.shape[1]
    #参数更新次数
    for it in range(iters):
        error_val = np.dot(x, theta.T) - y
        error_val = np.reshape(error_val, (50,))
        update_val_temp = np.zeros(shape=x.shape)
        #参数个数
        for p in range(len_params):
            update_val_temp[:,p] = error_val * x[:,p]
        update_val = np.mean(update_val_temp, axis=0)
        #更新参数
        theta = theta - alpha * update_val
        print('第%d次训练===截距：%f,斜率%f' % (it, theta[0,0], theta[0,1]))
    cost = compute_cost(x, y, theta)
    return cost, theta

#初始化参数
#x插入一列值为1
x = np.reshape(x, (-1,1))
x = np.insert(x, 0, 1, axis=1)
#y值重新reshape一下
y = np.reshape(y, (-1,1))
theta = np.zeros(shape=(1, x.shape[1]))
#初始化参数
alpha = 0.01
iters = 2000
cost, theta_v = grandient_descent(x, y, theta, alpha, iters)
print(cost, theta_v)


# %%


#%%
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
#%%
'''创造数据X Y'''
reg_data, reg_target = make_regression(n_samples=200, n_features=10, n_informative=5, noise=5)
#%%
 
''' 通过交叉检验来获取最优参数'''
from sklearn.linear_model import LassoCV
lassocv = LassoCV()
lassocv.fit(reg_data, reg_target)
alpha = lassocv.alpha_
print('利用Lasso交叉检验计算得出的最优alpha：' + str(alpha))
 
 
'''lasso回归'''
lasso = Lasso(1)
lasso.fit(reg_data, reg_target)
 
 
'''计算系数不为0的个数'''
import numpy as np
n = np.sum(lasso.coef_ != 0)
print('Lasso回归后系数不为0的个数：' + str(n))
 
'''输出结果
   如果names没有定义，则用X1 X2 X3代替
   如果Sort = True，会将系数最大的X放在最前'''
def pretty_print_linear(coefs, names = None, sort = False):  
    if names == None:  
        names = ["X%s" % x for x in range(len(coefs))]  
    lst = zip(coefs, names)  
    if sort:  
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))  
    return " + ".join("%s * %s" % (round(coef, 3), name)  
                                   for coef, name in lst)  
 
 
 
print('Y = '+ pretty_print_linear(lasso.coef_))
