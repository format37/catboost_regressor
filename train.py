import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

data = pd.read_csv('in.txt', delimiter="\t")
data.columns = ["id", "product_group", "div", "fault_a", "fault_b", "work", "duration"]
data.sort_values(['duration'], inplace=True)
data.reset_index(drop=True, inplace=True)
bottom=len(data)//100*5
top=len(data)-bottom
data.drop(data.index[[i for i in range(top,len(data))]], inplace=True)
data.drop(data.index[[i for i in range(0,bottom)]], inplace=True)
data.drop(data.columns[0], axis=1, inplace=True)
data.fillna(0,inplace=True)
data=data.sample(n=len(data))
msk = np.random.rand(len(data)) < 0.75
train_df = data[msk]
test_df = data[~msk]
X = train_df.drop('duration', axis=1)
y=train_df.duration
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
X_test = test_df
model1 = CatBoostRegressor(iterations=2,
                           learning_rate=0.2,
                           depth=2)
model1.fit(X_train, y_train)
model1.fit(X_train, y_train, init_model=model1)
model2 = CatBoostRegressor(iterations=4,
                           learning_rate=0.1,
                           depth=4)
model2.fit(X_train, y_train, init_model=model1)
model2.save_model('model.dump')