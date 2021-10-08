from aiohttp import web
import os
import pandas as pd
from io import StringIO
from catboost import CatBoostRegressor

async def call_test(request):
	content = "get ok"
	return web.Response(text=content,content_type="text/html")


async def call_inference(request):

	response = 'ok'

	"""хттпЗапрос	= Элемент.АдресСкриптаОбучения
			+"?model="+Элемент.Модель			
			+"&request="+ид_запроса
			+"&iter_count="+Элемент.КоличествоИтераций
			+"&learning_rate="+Формат(Элемент.КоэффициентОбучения,"ЧРД=.; ЧН=; ЧГ=")
			+"&depth="+Элемент.ГлубинаОбуения
			+"&data_file="+Элемент.ФайлДанных
			+"&cat_features="+КатегориальныеПризнаки;
		А_Серверные.ВыполнитьHTTPЗапросПолучитьОтвет(хттпЗапрос,Ответ,Элемент.ТаймаутЗапроса,Ложь);"""
	
	# read csv request as pandas df
	csv_text = str(await request.text()).replace('\ufeff', '')
	df = pd.read_csv(StringIO(csv_text), sep=';')

	#debug save to file
	df.to_csv('volume/inference_in.csv')

	# model name	
	modelName=df.model.iloc()[0]
	df.drop('model', axis=1, inplace=True)
		
	# query="select row,field,value from regression where request='"+request+"' order by row,field;"

	#read
	# data_source = pd.read_sql(query, con=con)

	"""fields_count=max(df['field'])
	new_columns=tuple('field_'+str(i) for i in range(1,int(fields_count)+1))
	data = pd.DataFrame([])"""

	"""#predict		
	model = CatBoostRegressor()
	model.load_model(modelName)
	pred = model.predict(data)
	#insert
	df = pd.DataFrame({'pred':pred})
	df.columns = ['value']"""

	"""#save to sql
	rows=df[df.field==0].row.reset_index(drop=True)
	predict_field=max(df.field)+1
	sql_df=pd.DataFrame({
		'row':rows,
		'field':[predict_field for i in range(0,len(rows))],
		'value':df.value,
		'request':[request for i in range(0,len(rows))]
	})
	engine = create_engine('mssql+pymssql://'+username+':'+password+'@'+ServerName+'/'+Database)
	sql_df.to_sql('regression', con=engine, if_exists='append', index=False)"""

	# return web.Response(text=content,content_type="text/html")


	return web.Response(text=str(response),content_type="text/html")

app = web.Application(client_max_size=1024**3)
app.router.add_route('GET', '/test', call_test)
app.router.add_post('/inference', call_inference)

web.run_app(
    app,
    port=os.environ.get('PORT', ''),
)
