import pandas as pd
import datetime as dt
from tqdm import tqdm
import numpy as np
import networkx as nx
# import warnings
import os
# warnings.filterwarnings("ignore")

week = int(input("Input period size (weeks): "))
windowed = input("Input windowed (y/n): ")
windowed = windowed == "y"

#################### LOAD DATA ############################

print('Reading data...')

sample = pd.read_csv('/home/pelle/Master_Thesis/data/raw/wallstreetbets/comments_pmaw_2016-2021_wsb.csv',nrows=10)
dtypes = sample.dtypes # Get the dtypes
cols = sample.columns # Get the columns
dtype_dictionary = {} 
for c in cols:
    if str(dtypes[c]) == 'int64':
        dtype_dictionary[c] = 'float32' # Handle NANs in int columns
    else:
        dtype_dictionary[c] = str(dtypes[c])
dtype_dictionary['author'] = 'str'
dtype_dictionary['body'] = 'str'
dtype_dictionary['parent_id'] = 'str'
dtype_dictionary['link_id'] = 'str'
dtype_dictionary['id'] = 'str'

df_comments_chunked = pd.read_csv('/home/pelle/Master_Thesis/data/raw/wallstreetbets/comments_pmaw_2016-2021_wsb.csv', dtype=dtype_dictionary, 
    on_bad_lines='skip',
    chunksize=1000000,
    low_memory=False,
    usecols=['author','parent_author','created_utc','score'],
    dtype=dtype_dictionary,
    )

df_comments = pd.concat(df_comments_chunked, ignore_index=True)
df_comments.created_utc = pd.to_datetime(df_comments.created_utc,unit='s')

print('Done reading!'+'\n')

######################### GRAPHS CONSTRUCTION #########################

folder_name_end = 'graphs_windowed_' + str(week) + '/' if windowed else 'graphs_' + str(week) + '/'
folder_name = '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/' + folder_name_end
print('folder_name: ', folder_name+'\n')

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print('Created folder: ', folder_name)
else:
    print('Folder already exists: ', folder_name)
    print('Overwriting files in folder..')


start = df_comments.created_utc.min()
period = df_comments.created_utc.max()-df_comments.created_utc.min()
# start = dt.datetime.strptime('2018-11-1', '%Y-%m-%d')
# period = dt.datetime.strptime('2019-11-1', '%Y-%m-%d') - dt.datetime.strptime('2018-11-1', '%Y-%m-%d')
step = dt.timedelta(weeks=week)

print('start: ', start)
print('period: ', period)
print('step: ', step)
if windowed:
    print('Windowed')
    for i in tqdm(range(int(np.ceil(period / step)))):
        end = start + step*i + step
        begin = start + step*i
        df_comments_period = df_comments[(df_comments.created_utc < end) & (df_comments.created_utc > begin)]

        G = nx.Graph()
        G.add_edges_from(df_comments_period[['author','parent_author']].dropna().values)

        file_name = 'graph_' + str(begin)[:10] + '_' + str(end)[:10] + '.gpickle'
        nx.write_gpickle(G, folder_name + file_name)

else:
    print('Not windowed')
    for i in tqdm(range(int(np.ceil(period / step)))):
        end = start + step*i + step
        df_comments_before=df_comments.loc[df_comments.created_utc<end]
        
        G = nx.Graph()
        G.add_edges_from(df_comments_before[['author','parent_author']].dropna().values)

        file_name = 'graph_' + str(end)[:10] + '.gpickle'
        nx.write_gpickle(G, folder_name + file_name)

print('Done!')

# to do:
# - add weights to edges
# - add directed edges