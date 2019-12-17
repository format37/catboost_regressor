#!/usr/bin/env python3
print("Content-type: text/html")
print()
import pandas as pd
from catboost import CatBoostRegressor
path='/mnt/share/1c_map_uid/'
uid='1edb6d74-b889-4082-a27f-9580eacca2e6'
data = pd.read_csv( path + uid + '.txt', delimiter="\t")
data.columns = ["product_group", "div", "fault_a", "fault_b", "work"]
model = CatBoostRegressor()
model.load_model('cgi-bin/model.dump')
pred = model.predict(data, 
        ntree_start=0, 
        ntree_end=0, 
        thread_count=-1,
        verbose=None)
df = pd.DataFrame({'pred':pred})
#df.to_csv(path + uid + '_out' + '.txt', sep=';', encoding='utf-8')
print(df)
#print("Hell_o")
