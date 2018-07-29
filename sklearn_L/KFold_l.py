from sklearn.model_selection import KFold  
import numpy as np  
X = np.arange(24).reshape(12,2)  
y = np.random.choice([1,2],12,p=[0.4,0.6])  
print(X,'\n',y)
kf = KFold(n_splits=5,shuffle=False)  
for train_index , test_index in kf.split(X):  
    print('train_index:%s , test_index: %s ' %(train_index,test_index))  