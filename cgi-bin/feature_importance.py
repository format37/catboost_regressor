#!/usr/bin/env python3
#http://10.2.4.87:8000/cgi-bin/feature_importance.py?model=cgi-bin/composite_test_bid
import cgi
import pymysql.cursors
import pandas as pd
from catboost import CatBoostRegressor

print("Content-type: text/html")
print()

form = cgi.FieldStorage(keep_blank_values=1)
modelName=form['model'].value
model = CatBoostRegressor()
model.load_model(modelName)

fi=[i for i in model.get_feature_importance(type="FeatureImportance")]
for i in range(len(fi)):
	print('field_'+str(i+1)+','+str(fi[i]))