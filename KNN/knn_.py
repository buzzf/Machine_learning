
# 房价预测

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 读取数据,数据预处理

features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
dc_listings = pd.read_csv('listings.csv')
dc_listings = dc_listings[features]
dc_listings['price'] = dc_listings.price.str.replace('\$|,', '').astype(float)
dc_listings = dc_listings.dropna()
dc_listings[features] = StandardScaler().fit_transform(dc_listings[features])
normallized_list = dc_listings
# print(normallized_list.head())

train_df = normallized_list.copy().iloc[:2792]
test_df = normallized_list.copy().iloc[2792:]


# 单因子预测
# 预测三个房间的价格

def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(normallized_list[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price, feature_column='accommodates')
# 模型评估 RMSE

test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)


# 多变量KNN模型
# scipy中有现成的距离计算公式
def predict_price_multivariate(new_listing_value,feature_columns):
    temp_df = train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing_value[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

cols = ['accommodates', 'bathrooms']
test_df['predicted_price'] = test_df[cols].apply(predict_price_multivariate,feature_columns=cols, axis=1)    
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)


# ××××××××××××××××××××××××××××× sklearn ××××××××××××××××××××××××××
# 使用sklean来完成KNN


cols = ['accommodates','bedrooms','bathrooms','beds','minimum_nights','maximum_nights','number_of_reviews']
knn = KNeighborsRegressor()
knn.fit(train_df[cols], train_df['price'])
predictions = knn.predict(test_df[cols])
# print(predictions)

# 模型评估
mse = mean_squared_error(test_df['price'], predictions)
rmse = mse ** (1/2)
print(rmse)