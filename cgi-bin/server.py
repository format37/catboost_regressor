#!/usr/bin/env python
import asyncio
from aiohttp import web
import urllib
import requests
import datetime
import cgi
import pymssql
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from catboost.utils import eval_metric
from sklearn.model_selection import train_test_split
import time
from sqlalchemy import create_engine

with open('/home/alex/projects/1c_ml_regression_diagnostics/cgi-bin/telegram.group','r') as fh:
    telegram_group=fh.read()
    fh.close()

def sql_params():
    ServerName='10.2.4.124'
    Database='1c_python'
    username = 'ICECORP\\1csystem'
    password = '0dKasn@ms+'
    return ServerName, Database, username, password

def send_to_telegram(chat,message):
    headers = {
        "Origin": "http://scriptlab.net",
        "Referer": "http://scriptlab.net/telegram/bots/relaybot/",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'
        }
    url     = "http://scriptlab.net/telegram/bots/relaybot/relaylocked.php?chat="+chat+"&text="+urllib.parse.quote_plus(message)
    return requests.get(url,headers = headers)

async def predict(request):

    content = ''

    try:
        modelName="/home/alex/projects/1c_ml_regression_diagnostics/cgi-bin/"+request.rel_url.query['model']	
    except Exception as e:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' predict error: "model" parameter not received')

    try:
        request=request.rel_url.query['request']	
    except Exception as e:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' predict error: "request" parameter not received')

    #load data from sql
    ServerName, Database, username, password = sql_params()	
    con	= pymssql.connect(ServerName, username, password, Database)

    with con:

        cur = con.cursor()
        query="select row,field,value from regression where request='"+request+"' order by row,field;"
        cur.execute(query)

        #read
        data_source = pd.read_sql(query, con=con)

        #debug save to file
        data_source.to_csv('/home/alex/projects/1c_ml_regression_diagnostics/cgi-bin/data/pred.csv')

        fields_count=max(data_source['field'])
        new_columns=tuple('field_'+str(i) for i in range(1,int(fields_count)+1))
        data = pd.DataFrame([])

        #rotate
        for row in range (0,max(data_source['row'])+1):
            temp=pd.DataFrame(data_source[data_source.row==row]['value']).transpose()
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

    return web.Response(text=content,content_type="text/html")

async def train(request):
    content = ''

    try:
        modelName="/home/alex/projects/1c_ml_regression_diagnostics/cgi-bin/"+request.rel_url.query['model']	
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "model" parameter not received')

    try:
        request_id=request.rel_url.query['request']	
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "request" parameter not received')

    try:
        in_iter_count=int(request.rel_url.query['iter_count'])
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "iter_count" parameter not received')

    try:
        in_learning_rate=float(request.rel_url.query['learning_rate'])
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "learning_rate" parameter not received')

    try:
        in_depth=request.rel_url.query['depth']	
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "depth" parameter not received')

    try:
        in_data_file=request.rel_url.query['data_file']	
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "data_file" parameter not received')

    try:
        cat_features=request.rel_url.query['cat_features']	
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "cat_features" parameter not received')

    step = 0

    try:

        if len(cat_features)==0:
            cat_features	= None
        else:
            cat_features	= cat_features.split(',')
        target='field_0'

        #load data from sql
        ServerName, Database, username, password = sql_params()			
        con	= pymssql.connect(ServerName, username, password, Database)

        step+=1

        with con:

            cur = con.cursor()
            query="select row,field,value from regression where request='"+request_id+"' order by row,field;"
            cur.execute(query)
            data_source = pd.read_sql(query, con=con)
            if len(in_data_file)>0:
                data_source.to_csv(in_data_file)
            fields_count=max(data_source['field'])+1
            columns=tuple('field_'+str(i) for i in range(0,int(fields_count)))
            data = pd.DataFrame([])
            for row in range (0,max(data_source['row'])+1):
                temp=pd.DataFrame(data_source[data_source.row==row]['value']).transpose()
                temp.columns=columns
                data = data.append(temp,ignore_index = True)
            step+=1

            data.sort_values(['field_0'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.fillna(0,inplace=True)
            data=data.sample(n=len(data))

            #msk = np.random.rand(len(data)) < 0.75
            #train_df = data[msk]
            #test_df = data[~msk]
            #X = train_df.drop('field_0', axis=1)
            #y=train_df.field_0
            #X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, #random_state=42)

            X=data.drop('field_0', axis=1)
            y=data.field_0
            X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)			
            content+='y: '+str(len(y))+'\n'
            content+='y_train: '+str(len(y_train))+'\n'
            content+='y_validation: '+str(len(y_validation))+'\n'
            step+=1

            #X_test = test_df.drop('field_0', axis=1)
            eval_dataset = Pool(data=X_validation,
                    label=y_validation,
                    cat_features=cat_features)

            # model1 = CatBoostRegressor(iterations=in_iter_count,
                                       # learning_rate=in_learning_rate,
                                       # depth=int(in_depth),
                                       # cat_features=cat_features,
                                       # metric_period=int(in_iter_count/10) if int(in_iter_count/10)>0 else 1
                                       # )
            model1 = CatBoostRegressor(
                cat_features=cat_features,
                boost_from_average=True,
                score_function = 'NewtonL2',
                one_hot_max_size = 512,
                depth = 16,
                langevin = True,
                posterior_sampling=True,
                model_shrink_rate = 1/(2*len(y_train)),
                verbose=False
                )
            #model1.fit(X_train, y_train)
            model1.fit(X_train, y_train,use_best_model = True,eval_set=eval_dataset)
            pred = model1.predict(X_validation)			
            params = model1.get_params()
            content+=str(params)+'\n'
            content+='\n'+params['loss_function']+' loss: '+ str(eval_metric(y_validation.to_numpy(), pred, params['loss_function']))
            content+='\nFitted: '+str(model1.is_fitted())
            content+='\nModel score:\n'+str(model1.score(X,y))
            step+=1
            model1.save_model(modelName)
            query="delete from regression where request='"+request_id+"';"
            cur.execute(query)

    except Exception as e:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: step '+str(step)+'\n'+str(e))

    return web.Response(text=content,content_type="text/html")

async def importance(request):

    print(str(datetime.datetime.now())+' importance requested')

    content = ''

    try:
        modelName="/home/alex/projects/1c_ml_regression_diagnostics/cgi-bin/"+request.rel_url.query['model']
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' train error: "model" parameter not received')

    try:

        model = CatBoostRegressor()
        model.load_model(modelName)

        fi=[i for i in model.get_feature_importance(type="FeatureImportance")]
        for i in range(len(fi)):
            content+=('' if len(content)==0 else ' ')+'field_'+str(i+1)+','+str(fi[i])
    except:
        send_to_telegram(telegram_group,str(datetime.datetime.now())+' feature importance call error')

    return web.Response(text=content,content_type="text/html")

app = web.Application()
app.router.add_route('GET', '/predict_ml', predict)
app.router.add_route('GET', '/train_ml', train)
app.router.add_route('GET', '/feature_importance', importance)

send_to_telegram(telegram_group,str(datetime.datetime.now())+' ml server started')

loop = asyncio.get_event_loop()
handler = app.make_handler()
f = loop.create_server(handler, port='8000')
srv = loop.run_until_complete(f)

print('serving on', srv.sockets[0].getsockname())
try:
    loop.run_forever()
except KeyboardInterrupt:
    print("serving off...")
finally:
    loop.run_until_complete(handler.finish_connections(1.0))
    srv.close()
