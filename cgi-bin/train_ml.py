#!/usr/bin/env python3
# alter table regression convert to character set utf8mb4 collate utf8mb4_unicode_ci
# ALTER TABLE regression MODIFY value LONGTEXT
import cgi
import pymysql.cursors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

print("Content-type: text/html")
print()

# LOAD TABLE FROM SQL
ServerName='localhost'
Database='ml'
username = '1c'
password = 'mTh!^z!2dqy5d-.Rc7Yc'

con = pymysql.connect(ServerName, username, password, Database)
form = cgi.FieldStorage(keep_blank_values=1)

request=form['request'].value
modelName=form['model'].value
cat_features=form['cat_features'].value.split(',')
if len(cat_features)==0:
	cat_features=None

with con:
	cur = con.cursor()

	query="select row,field,value from regression where request='"+request+"' order by row,field;"
	cur.execute(query)
	data_source = pd.read_sql(query, con=con)
	fields_count=max(data_source['field'])+1
	columns=tuple('field_'+str(i) for i in range(0,int(fields_count)))
	data = pd.DataFrame([])
	for row in range (0,max(data_source['row'])+1):
		temp=pd.DataFrame(data_source[data_source.row==row]['value']).transpose()
		temp.columns=columns
		data = data.append(temp,ignore_index = True)

	data.sort_values(['field_0'], inplace=True)
	data.reset_index(drop=True, inplace=True)
	data.fillna(0,inplace=True)
	data=data.sample(n=len(data))
	msk = np.random.rand(len(data)) < 0.75
	train_df = data[msk]
	test_df = data[~msk]
	X = train_df.drop('field_0', axis=1)
	y=train_df.field_0
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
	print('y: ',len(y))
	print('y_train: ',len(y_train))
	print('y_validation: ',len(y_validation))
	X_test = test_df.drop('field_0', axis=1)
	model1 = CatBoostRegressor(iterations=2000,
							   learning_rate=0.01,
							   depth=16,
							   cat_features=cat_features,
							   metric_period=100
							   )
	model1.fit(X_train, y_train)
	print('best iteration: ',str(model1.get_best_iteration()))
	print('best score: ',str(model1.get_best_score()))
	print('final score: ')
	print(str(model1.score(X,y)))
	'''
	model2 = CatBoostRegressor(iterations=1000,
							   learning_rate=0.1,
							   depth=4,
							   cat_features=cat_features)
	model2.fit(X_train, y_train, init_model=model1)
	'''
	model1.save_model(modelName)
	
	res = model.calc_feature_statistics(X_train,
                                    y_train,
                                    feature=4,
                                    plot=True)
	
	query="delete from regression where request='"+request+"';"
	cur.execute(query)
	
	file = open('train.log', 'w')