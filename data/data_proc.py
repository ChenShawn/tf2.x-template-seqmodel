import pandas as pd
from tqdm import tqdm
import datetime
import calendar
import numpy as np
import json


print('>>>>>>>> STEP 1: Feature preprocessing')
df = pd.read_csv('./data.csv')
sorted_df = df.sort_values(by=['amount', 'diff'], ascending=False)

dirdict = {
    '进港': 'in',
    '出港': 'out'
}

def is_weekend(daystr):
    daynums = list(map(int, daystr.split('-')))
    date = datetime.date(*daynums)
    return 'yes' if date.weekday() == 5 or date.weekday() == 6 else 'no'

output_df = sorted_df.copy()
output_df['is_weekend'] = sorted_df['takeoffdate'].apply(is_weekend)
output_df['year'] = sorted_df['takeoffdate'].apply(lambda x: float(x.split('-')[0]))
output_df['month'] = sorted_df['takeoffdate'].apply(lambda x: float(x.split('-')[1]))
output_df['day'] = sorted_df['takeoffdate'].apply(lambda x: float(x.split('-')[2]))
output_df['cityid'] = sorted_df['cityid'].apply(lambda x: '_' + str(x))
output_df['direction'] = sorted_df['direction'].apply(lambda x: dirdict[x])

output_df = output_df.sort_values(by=['takeoffdate'])
output_df.to_csv('single_line.csv')

"""
## Data preprocessing

- Divide categorical columns and numerical columns
- Seperate them into two dict
- Categorical columns: list their vocabulary list
- Numerical columns: list their mean and stddev values
"""

print('>>>>>>>> STEP 2: Generate info.json')
EPSILON = 1e-6

catcols, numcols = {}, {}
COLUMNS = output_df.columns
for col in COLUMNS:
    if output_df[col].dtype == np.int64 or output_df[col].dtype == np.float64:
        col_std = output_df[col].std()
        if col_std < EPSILON:
            col_std = EPSILON
        numcols[col] = [output_df[col].mean(), col_std]
    else:
        catcols[col] = list(set(output_df[col].values.tolist()))
        
with open('info.json', 'w') as fd:
    jsoninfo = {'catcols': catcols, 'numcols': numcols}
    json.dump(jsoninfo, fd, allow_nan=False, indent=2)


print('>>>>>>>> STEP 3: split train set and validation set')
# 1. seperate train/eval
# 2. normalize labels
df = pd.read_csv('single_line.csv')
with open('./info.json', 'r') as fd:
    jsoninfo = json.load(fd)
label_names = ['amount', 'avg_loadfactor']

for label in label_names:
    df[label] = df[label].apply(lambda x: (x - jsoninfo['numcols'][label][0]) / jsoninfo['numcols'][label][1])
    
datasize = len(df)
eval_size = datasize // 5
train_size = datasize - eval_size
print('[*] data_size={} train_size={} eval_size={}'.format(datasize, train_size, eval_size))

df_train = df.iloc[: train_size]
df_eval = df.iloc[train_size: ]

df_train.to_csv('./single_line_train.csv')
df_eval.to_csv('./single_line_test.csv')

print('>>>>>>>>>>>>>>>>>>> Done!!!')
