import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


'''
(features): average income,
housing average age, average rooms, average bedrooms, population,
average occupation, latitude, and longitude
'''

housing = fetch_california_housing()
# print(housing.data[0])


# 可视化
def visual_tree(housing):
    dtr = tree.DecisionTreeRegressor(max_depth=2)
    dtr.fit(housing.data[:, [6,7]], housing.target)
    dot_data = tree.export_graphviz(
            dtr,
            out_file = None,
            feature_names = housing.feature_names[6:8],
            filled = True,
            impurity = False,
            rounded = True
        )

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    graph.write_png('dtr_tree.png')


data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)

def dTreeRegressor(data_train, target_train, data_test, target_test):
    dtr = tree.DecisionTreeRegressor(random_state = 42)
    dtr.fit(data_train, target_train)
    score = dtr.score(data_test, target_test)
    return score

# 交叉验证

tree_param_grid = { 'min_samples_split': list((3,6,9)) ,'n_estimators':list((10,50,100))}
grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print(grid.cv_results_ , grid.best_params_)