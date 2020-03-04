#!/usr/bin/env python3
import cgi
import pymssql
import pandas as pd
from catboost import CatBoostRegressor
import datetime
import time
from sqlalchemy import create_engine

def log(log_file,message):
	log_file.write( str(datetime.datetime.now()) +" - "+str(message)+"\n" )

print("Content-type: text/html")
print()

# LOAD TABLE FROM SQL
ServerName='10.2.4.25'
Database='1c_python'
username = 'ICECORP\\1csystem'
password = '0dKasn@ms+'

with open("logs/log-"+str(time.strftime("%Y%m%d-%H%M%S"))+".txt", "w") as log_file:

	log(log_file,'connect')
	con	= pymssql.connect(ServerName, username, password, Database)
	form = cgi.FieldStorage(keep_blank_values=1)
	modelName=form['model'].value
	request=form['request'].value

	with con:
	
		log(log_file,'select')
		cur = con.cursor()
		query="select row,field,value from regression where request='"+request+"' order by row,field;"
		cur.execute(query)		
		log(log_file,'pd read')		
		data_source = pd.read_sql(query, con=con)
		fields_count=max(data_source['field'])
		new_columns=tuple('field_'+str(i) for i in range(1,int(fields_count)+1))
		data = pd.DataFrame([])
		log(log_file,'rotate')
		for row in range (0,max(data_source['row'])+1):
			temp=pd.DataFrame(data_source[data_source.row==row]['value']).transpose()
			temp=temp.drop(temp.columns[0], axis=1)
			temp.set_axis(new_columns, axis='columns', inplace=True)
			data = data.append(temp,ignore_index = True)
		log(log_file,'predict')
		
		# PREDICT
		model = CatBoostRegressor()
		model.load_model(modelName)
		pred = model.predict(data, 
				ntree_start=0, 
				ntree_end=0, 
				thread_count=-1,
				verbose=True)
		log(log_file,'insert')
		df = pd.DataFrame({'pred':pred})
		df.columns = ['value']
				
		#SAVE TO SQL
		rows=data_source[data_source.field==0].row.reset_index(drop=True)
		predict_field=max(data_source.field)+1
		sql_df=pd.DataFrame({
			'row':rows,
			'field':[predict_field for i in range(0,len(rows))],
			'value':df.value,
			'request':[request for i in range(0,len(rows))]
		})
		engine = create_engine('mssql+pymssql://'+username+':'+password+'@'+ServerName+'/'+Database)
		sql_df.to_sql('regression', con=engine, if_exists='append', index=False)
		log(log_file,'complete')
		exit()