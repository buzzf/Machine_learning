import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('iris.data')
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
X = df.ix[:,0:4].values
y = df.ix[:,4].values
X_std = StandardScaler().fit_transform(X)


cov_mat = np.cov(X_std.T)  # 求协方差矩阵
eig_vals, eig_vecs = np.linalg.eig(cov_mat) # 特征值和特征向量
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))] # 特征值和特征向量对应
eig_pairs.sort(key=lambda x: x[0], reverse=True)  # 根据特征值排序

# 计算各个特征值影响占比,决定降成几维，比如大于95%
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp) # 求累加和 [1,2,3] --> [1,3,6]
print(cum_var_exp)


# 降维，原X是 n*4,要转化为n*2, 中间矩阵为4*2，则取出前两个特征向量组合在一起就好了
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

X_pca = X_std.dot(matrix_w)  # n*2矩阵
print(X_std[:5])
print(X_pca[:5])