
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing

'''
我们将建立一个逻辑回归模型来预测一个学生是否被大学录取。假设你是一个大学系的管理员，你想根据两次考试的结果来决定每个申请人的录取机会。你有以前的申请人的历史数据，你可以用它作为逻辑回归的训练集。对于每一个培训例子，你有两个考试的申请人的分数和录取决定。为了做到这一点，我们将建立一个分类模型，根据考试成绩估计入学概率。

'''


# 数据读取与预处理

pdData = pd.read_csv('./data/LogiReg_data.txt', header=None, names=['exam1', 'exam2', 'admitted'])
pdData.insert(0, 'ones', 1)
orig_data = pdData.as_matrix()
theta = np.zeros([1, orig_data.shape[1]-1])  # 构建 h(x) = theta.dot(x)
# print(X[:5])

def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

# 建立分类器

def sigmoid(z):
	return 1/(1+np.exp(-z))

def model(X, theta):
	return sigmoid(np.dot(X, theta.T))

# cost function  D(hθ(x),y)= −ylog(hθ(x))−(1−y)log(1−hθ(x))   J(θ) = 1/n ∑(hθ(xi),yi)
def cost(X, y, theta):
	left = np.multiply(-y, np.log(model(X, theta)))
	right = np.multiply(1-y, np.log(1-model(X, theta)))
	return np.sum(left - right)/len(X)

# gradient 
def gradient(X, y, theta):
	grad = np.zeros(theta.shape)
	error = (model(X, theta)-y).ravel()
	for j in range(len(theta.ravel())):
		term = np.multiply(error, X[:,j])
		grad[0, j] = np.sum(term)/len(X)
	return grad

# Gradient descent  比较3中不同梯度下降方法
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold  # 线性代数中的范数

def descent(data, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解
    
    init_time = time.time()
    i = 0 # 迭代次数
    k = 0 # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape) # 计算的梯度
    costs = [cost(X, y, theta)] # 损失值

    
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize #取batch数量个数据
        if k >= n: 
            k = 0 
            X, y = shuffleData(data) #重新洗牌
        theta = theta - alpha*grad # 参数更新
        costs.append(cost(X, y, theta)) # 计算新的损失
        i += 1 

        if stopType == STOP_ITER:       value = i
        elif stopType == STOP_COST:     value = costs
        elif stopType == STOP_GRAD:     value = grad
        if stopCriterion(stopType, value, thresh): 
        	break
    
    return theta, i-1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: 
    	strDescType = "Gradient"
    elif batchSize==1:  
    	strDescType = "Stochastic"
    else: 
    	strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: 
    	strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: 
    	strStop = "costs change < {}".format(thresh)
    else: 
    	strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    fig.savefig('./costpic/{}\nTheta:{}-Iter:{}-Last_cost:{:03.2f}-Duration:{:03.2f}s.png'.format(
        name, theta, iter, costs[-1], dur))
    return theta

n=100
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)   # 根据迭代次数停止
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)  # 根据损失值停止
runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)     # 根据梯度变化停止


# # 对比不同的梯度下降方法
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001) # Stochastic descent
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)  # Stochastic descent
# runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)   # Mini-batch descent

# 将数据标准化
scaled_data = orig_data.copy()
scaled_data[:, 1:3] = preprocessing.scale(orig_data[:, 1:3])
# runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)
# runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)
# runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
# runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)