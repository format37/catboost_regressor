#!/usr/bin/env python3
import cgi
import pymysql.cursors
import pandas as pd
from catboost import CatBoostRegressor

#http://10.2.4.87:8000/cgi-bin/predict.py?model=model.dump&field_count=5&row_count=7&request=9bfd0165-a182-4372-8b49-91ef225aa991

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

	# PREDICT
	model = CatBoostRegressor()
	model.load_model('cgi-bin/diagnostics_duration_all.dump')
	pred = model.predict(data, 
			ntree_start=0, 
			ntree_end=0, 
			thread_count=-1,
			verbose=None)
	df = pd.DataFrame({'pred':pred})
	df.columns = ['value']

	#SAVE TO SQL
	query="insert into regression(request,row,field,value) values "
	first=True
	row=0
	for record in df.values:
		value=record[0]
		field=field_count
		query+=("" if first else ",")+"('"+request+"',"+str(row)+","+field+",'"+str(value)+"')"
		row+=1
		first=False
	query+=';'
	#print(query)
	cur.execute(query)
	con.commit()
	con.close()
	print('complete')