import pandas as pd
from catboost import CatBoostRegressor
data = pd.read_csv('in.txt', delimiter="\t")
data.columns = ["id", "product_group", "div", "fault_a", "fault_b", "work", "duration"]
data = data[data.columns.drop(['id','duration'])]
model = CatBoostRegressor()
model.load_model('diagnostics_duration_model.dump');
pred = model.predict(data, 
        ntree_start=0, 
        ntree_end=0, 
        thread_count=-1,
        verbose=None)
df = pd.DataFrame({'pred':pred})
df.to_csv('out.txt', sep='\n', encoding='utf-8')