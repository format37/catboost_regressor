from aiohttp import web
import os
import pandas as pd
from io import StringIO
from catboost import CatBoostRegressor

async def call_test(request):
	content = "get ok"
	return web.Response(text=content,content_type="text/html")


async def call_inference(request):
	
	# read csv request as pandas df
	csv_text = str(await request.text()).replace('\ufeff', '')
	df = pd.read_csv(StringIO(csv_text), sep=';')

	#debug save to file
	df.to_csv('data/inference_in.csv')

	# model name	
	modelName=df.model.iloc()[0]
	df.drop('model', axis=1, inplace=True)
		
	# query="select row,field,value from regression where request='"+request+"' order by row,field;"

	#read
	# data_source = pd.read_sql(query, con=con)

	fields_count=max(df['field'])
	new_columns=tuple('field_'+str(i) for i in range(1,int(fields_count)+1))
	data = pd.DataFrame([])

	#rotate
	for row in range (0,max(df['row'])+1):
		temp=pd.DataFrame(df[df.row==row]['value']).transpose()
		temp=temp.drop(temp.columns[0], axis=1)
		temp.set_axis(new_columns, axis='columns', inplace=True)
		data = data.append(temp,ignore_index = True)

	#predict		
	model = CatBoostRegressor()
	model.load_model(modelName)
	pred = model.predict(data)
	#insert
	df = pd.DataFrame({'pred':pred})
	df.columns = ['value']

	#save to sql
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

	# return web.Response(text=content,content_type="text/html")


	return web.Response(text=str(response),content_type="text/html")

app = web.Application(client_max_size=1024**3)
app.router.add_route('GET', '/test', call_test)
app.router.add_post('/inference', call_inference)

web.run_app(
    app,
    port=os.environ.get('PORT', ''),
)
