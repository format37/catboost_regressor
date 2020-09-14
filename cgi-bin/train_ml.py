#!/usr/bin/env python

import requests
import urllib

def send_to_telegram(chat,message):
    try:
        #print('Telegram:',message)
        headers = {
            "Origin": "http://scriptlab.net",
            "Referer": "http://scriptlab.net/telegram/bots/relaybot/",
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'
            }
        url     = "http://scriptlab.net/telegram/bots/relaybot/relaylocked.php?chat="+chat+"&text="+urllib.parse.quote_plus(message)
        return requests.get(url,headers = headers)
    except Exception as e:
        return str(e)

try:

    import cgi
    #import pymysql.cursors #MySql
    import pymssql			#MsSql
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from catboost import CatBoostRegressor

    print("Content-type: text/html")
    print()
    print('start')
    exit()
    # LOAD TABLE FROM SQL
    ServerName='10.2.4.124'
    Database='1c_python'
    username = 'ICECORP\\1csystem'
    password = '0dKasn@ms+'

    #con = pymysql.connect(ServerName, username, password, Database)
    con	= pymssql.connect(ServerName, username, password, Database)
    form = cgi.FieldStorage(keep_blank_values=1)

    modelName		= form['model'].value
    request			= form['request'].value
    in_iter_count	= int(form['iter_count'].value)
    in_learning_rate= float(form['learning_rate'].value)
    in_depth		= int(form['depth'].value)
    in_data_file	= form['data_file'].value
    cat_features	= form['cat_features'].value

    if len(cat_features)==0:
        cat_features	= None
    else:
        cat_features	= cat_features.split(',')

    with con:
        cur = con.cursor()
        query="select row,field,value from regression where request='"+request+"' order by row,field;"
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
        model1 = CatBoostRegressor(iterations=in_iter_count,
                                   learning_rate=in_learning_rate,
                                   depth=in_depth,
                                   cat_features=cat_features,
                                   metric_period=int(in_iter_count/10) if int(in_iter_count/10)>0 else 1
                                   )
        model1.fit(X_train, y_train)
        print('final score: ')
        print(str(model1.score(X,y)))
        model1.save_model(modelName)
        query="delete from regression where request='"+request+"';"
        cur.execute(query)
    
except Exception as e:    
    send_to_telegram('106129214',str(datetime.datetime.now())+' service ml error: '+str(e))