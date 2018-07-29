import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv('iris.data')
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
X = df.ix[:,0:4].values
y = df.ix[:,4].values
# label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
enc = LabelEncoder()
y = enc.fit_transform(y) + 1

# 分别求三种鸢尾花数据在不同特征维度上的均值向量 mi
mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))

# 计算两个 4×4 维矩阵：类内散布矩阵和类间散布矩阵
# 类内散布矩阵
S_W = np.zeros((4,4))
for cl,mv in zip(range(1,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)


# 类间散布矩阵
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) # make column vector
    overall_mean = overall_mean.reshape(4,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('between-class Scatter Matrix:\n', S_B)


# 求解矩阵 S_W逆*S_B 的特征值
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))


# 特征值与特征向量：
# 特征向量：表示映射方向
# 特征值：特征向量的重要程度
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

# 特征值占比
print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

# 选前两维数据进行降维 n*4 --> n*2
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1))) # shape: 4*2
print('Matrix W:\n', W.real)
X_lda = X.dot(W)
print(X_lda[:5])


# ××××××××××××××××××××     SKlearn   ××××××××××××××××××××××
# 以上都是自己编写代码实现LDA降维
# 使用sklearn的LDA降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)
print(X_lda_sklearn.shape)
print(X_lda_sklearn[:5])