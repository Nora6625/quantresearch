# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:38:20 2020

@author: 15012
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from utils import *

dir_data_result = dir_main + "result/"

total_df = pd.read_csv(dir_data_final + "total_df_monthly.csv")

sgl_factors = pd.read_csv(dir_data_result + "sgl_factor_result.csv")

# 随机种子
random_seed = 42

"""
x为特征值，即各因子  y为收益率
"""

# 取出所选出来的单因子作为特征值
X = total_df[list(sgl_factors['Factor'])]
y = total_df['monthly_rtn']

# 处理NaN值：填充为特征列的均值
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# 处理无穷大和超出范围的值：将无穷大替换为NaN，然后填充为均值
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# 重新拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


rf = RandomForestRegressor(n_estimators=100, random_state=random_seed)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print('Mean Squared Error (MSE):', mse)
# print('Mean Absolute Error (MAE):', mae)
# print('R-squared (R2):', r2)


# 输出得到各因子的重要性

importance = rf.feature_importances_
print(importance)

for i in range(len(importance)):
    print(f"factor '{X.columns[i]}': {importance[i]}")

# 将每个月各股票得分保存到文件中

# 确保不会出现(12,)这种情况
IRs = np.array(sgl_factors['IR'])

weight = np.array(importance) * abs(IRs) / IRs

weight = weight.reshape(1, len(importance))
print(weight)

scores = X.dot(weight.T)
total_df['score'] = scores

print(total_df["score"])
# 得分是越低越好
total_df.to_csv(dir_data_result + "factor_scores.csv", index=False)