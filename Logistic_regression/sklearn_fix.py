import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from imblearn.over_sampling import SMOTE

'''
信用卡欺诈异常检测，0为正常，1为异常，分类任务

'''
# 数据读取，与预处理

data = pd.read_csv('逻辑回归-信用卡欺诈检测/creditcard.csv')
# print(data.head())
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# print(count_classes)
#  0    284315          可见样本极为不均衡：有两种方法处理：下采样：从多种选择与少中一样少
#  1       492                                        过采样：让少中扩展成与多一样多

# 绘图-分类统计
# count_classes.plot(kind='bar')
# plt.title("Fraud class histogram")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.savefig('class_count.png')

# Amount分布差异较大，需要进行标准化处理，另外Time列不需要
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time','Amount'],axis=1)
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

#×××××××××××××××× 下采样 ×××××××××××××××××××××

number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = data.iloc[under_sample_indices, :]
# print(under_sample_data.Class.value_counts())
X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

#  数据切分
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
    X_undersample,y_undersample, test_size=0.3, random_state=0)


# recall = TP/ TP + FN
# 交叉验证
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5,shuffle=False) # KFold将数据切分成5份
    c_param_range = [0.001,0.01,0.1,1,10,100]  # 正则化惩罚项
    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range
    
    j = 0
    for c_param in c_param_range:  # 找最好的正则化惩罚系数
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        m = 0
        for train_index, test_index in fold.split(x_train_data):  # 交叉验证

            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')
            lr.fit(x_train_data.iloc[train_index,:],y_train_data.iloc[train_index,:].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[test_index,:].values)
            recall_acc = recall_score(y_train_data.iloc[test_index,:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', m,': recall score = ', recall_acc)
            m += 1

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs) # 默认转化为object类型了
        results_table['Mean recall score'] = results_table['Mean recall score'].astype('float64')
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')
    
    # idxmax()可以返回数组中最大值的索引值
    best_c = results_table.iloc[results_table['Mean recall score'].idxmax()]['C_parameter']
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c

# best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)
# best_c2 = printing_Kfold_scores(X_train,y_train)


#×××××××××××××××× 过采样 ×××××××××××××××××××××
# SMOTE算法

features_train, features_test, label_train, label_test = train_test_split(X,y, test_size=0.3, random_state=0)
oversampler = SMOTE(random_state=0)
os_x,os_y = oversampler.fit_sample(features_train, label_train)  # 会自动平衡0与1的个数
# print(len(os_y[os_y==1]))
# print(len(os_y[os_y==0]))
os_x = pd.DataFrame(os_x)
os_y = pd.DataFrame(os_y)
best_c = printing_Kfold_scores(os_x, os_y)