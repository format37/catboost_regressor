#!/usr/bin/env python3
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
field_count=form['field_count'].value
request=form['request'].value
modelName=form['model'].value
integers=form['integers'].value.split(',')

with con:
	cur = con.cursor()

	for i in range(0,int(field_count)):
		query=" create temporary table f_"+str(i)
		query+=" select row,value from regression where request='"+request+"' and field="+str(i)+";"
		cur.execute(query)
		query=""

	query+=" create temporary table res"
	for i in range(0,int(field_count)):
		query+="" if i==0 else " union"
		query+=" select row"
		for j in range(0,int(field_count)):
			field_value = ' value' if i==j else ' null'
			query+=" ,"+field_value+" as field_"+str(j)
		query+=" from f_"+str(i)
	query+=";"
	cur.execute(query)
	query=""
	
	query+=" select "
	for i in range(0,int(field_count)):
		prefix="" if i==0 else ","
		query+=prefix+" max(field_"+str(i)+")"
	query+=" from res group by row order by row;"
	
	data = pd.read_sql(query, con=con)
	data.columns = tuple('field_'+str(i) for i in range(0,int(field_count)))
	for i in integers:	
		data['field_'+str(i)]	= pd.to_numeric(data['field_'+str(i)])
	'''
	for i in range(0,int(field_count)):	
		if str(i) in integers:
			data['field_'+str(i)]	= pd.to_numeric(data['field_'+str(i)])
		else:
			data['field_'+str(i)]	= data['field_'+str(i)].to_string()
	'''
	data.sort_values(['field_0'], inplace=True)
	data.reset_index(drop=True, inplace=True)
	bottom=len(data)//100*5
	top=len(data)-bottom
	data.drop(data.index[[i for i in range(top,len(data))]], inplace=True)
	data.drop(data.index[[i for i in range(0,bottom)]], inplace=True)
	data.fillna(0,inplace=True)
	data=data.sample(n=len(data))
	msk = np.random.rand(len(data)) < 0.75
	train_df = data[msk]
	test_df = data[~msk]
	X = train_df.drop('field_0', axis=1)
	y=train_df.field_0
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
	model2.save_model(modelName)
	
	query="delete from regression where request='"+request+"';"
	cur.execute(query)
	
	print(modelName)