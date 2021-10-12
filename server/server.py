from aiohttp import web
import os
import pandas as pd
from io import StringIO
from catboost import CatBoostRegressor, Pool
from catboost.utils import eval_metric
from sklearn.model_selection import train_test_split
import numpy as np

async def call_test(request):
	content = "ok"
	return web.Response(text=content,content_type="text/html")


async def call_train(request):

	response = ''

	# read csv request as pandas df
	csv_text = str(await request.text()).replace('\ufeff', '')
	df = pd.read_csv(StringIO(csv_text), sep=';')

	df.to_csv('data/in_train.csv') # TODO: remove this debug saver

	# read params
	first_row = df.iloc()[0]
	model_name = first_row.model
	cat_features = first_row.cat_features.split(',')
	print('cat_features', cat_features)
	
	# drop params columns
	df.drop([
		#'Unnamed: 0',
		'model',
		'cat_features'
	], axis=1, inplace=True)

	df.replace(np.nan, 'nan', inplace = True)

	# define dataset
	X = df.drop(df.columns[0], axis=1)
	y = df[df.columns[0]]
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)			
	response += 'y: '+str(len(y))+'\n'
	response += 'y_train: '+str(len(y_train))+'\n'
	response += 'y_validation: '+str(len(y_validation))+'\n'
	eval_dataset = Pool(data=X_validation,
			label=y_validation,
			cat_features=cat_features)

	
	# define model
	if os.environ.get('USE_GPU', '0') == '1':
		model = CatBoostRegressor(
			cat_features=cat_features,
			boost_from_average=True,
			#score_function = 'NewtonL2',
			one_hot_max_size = 256,
			depth = 16,
			langevin = True,
			#posterior_sampling=True,
			verbose=False,
			task_type="GPU"
			)
	else:
		model = CatBoostRegressor(
			cat_features=cat_features,
			boost_from_average=True,
			#score_function = 'NewtonL2',
			one_hot_max_size = 512,
			depth = 16,
			langevin = True,
			posterior_sampling=True,
			model_shrink_rate = 1/(2*len(y_train)),
			verbose=False,
			)

	# train
	model.fit(X_train, y_train, use_best_model=True, eval_set=eval_dataset)
	
	# evaluate model score
	pred = model.predict(X_validation)
	params = model.get_params()
	response += str(params)+'\n'
	response += '\n'+params['loss_function']+' loss: '+ str(eval_metric(y_validation.to_numpy(), pred, params['loss_function']))
	response += '\nFitted: '+str(model.is_fitted())
	response += '\nModel score:\n'+str(model.score(X,y))
	
	# save model
	model.save_model('data/'+model_name)

	return web.Response(text=str(response),content_type="text/html")


async def call_inference(request):
	# read csv request as pandas df
	csv_text = str(await request.text()).replace('\ufeff', '')
	df = pd.read_csv(StringIO(csv_text), sep=';')

	df.to_csv('data/inference_in.csv') # TODO: remove this debug saver

	# read params
	first_row = df.iloc()[0]
	model_name = first_row.model
	iterations_count = first_row.iterations_count
	learning_rate = first_row.learning_rate
	depth = first_row.depth

	# drop params columns
	df.drop([
		#'Unnamed: 0',
		'model',
		'iterations_count',
		'learning_rate',
		'depth'
	], axis=1, inplace=True)

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

	response = df.to_string()

	return web.Response(text=str(response),content_type="text/html")


def main():
	app = web.Application(client_max_size=1024**3)
	app.router.add_route('GET', '/test', call_test)
	app.router.add_post('/inference', call_inference)
	app.router.add_post('/train', call_train)

	web.run_app(
		app,
		port=os.environ.get('PORT', ''),
	)


if __name__ == "__main__":
    main()
