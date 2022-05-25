from aiohttp import web
import os
import pandas as pd
from io import StringIO
from catboost import CatBoostRegressor, Pool
from catboost.utils import eval_metric
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import numpy as np


async def call_test(request):
    content = "ok"
    return web.Response(text=content,content_type="text/html")


def prepare_data(csv_text, drop_outliers=True):
    df = pd.read_csv(StringIO(csv_text), sep=';')

    # debug save to csv
    #df.to_csv('data/in_inference.csv', sep=';')

    # read and drop params
    first_row = df.iloc()[0]
    model_name = first_row.model
    cat_features = first_row.cat_features.split(',')
    df.drop([
        'model',
        'cat_features'
    ], axis=1, inplace=True)

    df.replace(np.nan, 'nan', inplace = True)
    
    for column in cat_features:
        df[column] = df[column].astype(str)
    
    text_log = ''

    if drop_outliers:
        clf = IsolationForest(max_samples=100, random_state=42)
        clf.fit(df[df.columns[0]].values.reshape(-1, 1))
        y_noano = clf.predict(df[df.columns[0]].values.reshape(-1, 1))
        y_noano = pd.DataFrame(y_noano, columns=['Top'])
        y_noano[y_noano['Top'] == 1].index.values
        df = df.iloc[y_noano[y_noano['Top'] == 1].index.values]
        df.reset_index(drop=True, inplace=True)
        text_log+="\nNumber of Outliers:"+str(y_noano[y_noano['Top'] == -1].shape[0])
        text_log+="\nNumber of rows without outliers:"+str(df.shape[0])

    # define dataset
    X = df.drop(df.columns[0], axis=1)
    y = df[df.columns[0]]
    return df, X, y, cat_features, model_name, text_log


async def call_train(request):
    response = ''

    # read csv request
    csv_text = str(await request.text()).replace('\ufeff', '')
    
    # debug ++	
    #try:
    #with open('data/in_train.dat', 'w') as f:
    #	f.write(csv_text)
    #except Exception as e:
    #	print('debug save train data error:', str(e))
    # debug --

    df, X, y, cat_features, model_name, text_log = prepare_data(csv_text)

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
    response += text_log+'\n'
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
            one_hot_max_size = 128,
            depth = 16,
            langevin = True,
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
    response += '\nFeature importance:'
    try:
        importance = model.get_feature_importance()
        for i in range(len(model.feature_names_)):
            response += '\n'+str(np.round(importance[i],2)) + ' ' + model.feature_names_[i]
    except Exception as e:
        response += '\n'+str(e)       

    response = str(eval_metric(y_validation.to_numpy(), pred, params['loss_function'])[0]) + '\n' + response
    
    # save model
    model.save_model('data/'+model_name)

    return web.Response(text=str(response),content_type="text/html")


async def call_inference(request):
    response = ''

    # read csv request
    csv_text = str(await request.text()).replace('\ufeff', '')

    df, X, y, cat_features, model_name, log_txt = prepare_data(csv_text, False)

    # debug save to csv
    df.to_csv('data/out_inference.csv', sep=';')

    # predict
    model = CatBoostRegressor()
    model.load_model('data/'+model_name)
    df[df.columns[0]+'_predicted'] = model.predict(X)
    
    response  = df.to_csv(sep=';', index = False)

    return web.Response(text=response,content_type="text/html")


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
