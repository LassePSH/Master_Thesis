import pandas as pd
import datetime as dt
from tqdm import tqdm
import numpy as np
import networkx as nx



week = int(input("Input period size (weeks): "))
windowed = input("Input windowed (y/n): ")
windowed = windowed == "y"


# sample data to get dtypes
sample = pd.read_csv('/home/pelle/Downloads/comments_pmaw_2016-2021_wsb.csv',nrows=100)
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


# read data
print('Reading data...')
df_comments_chunked = pd.read_csv('/home/pelle/Downloads/comments_pmaw_2016-2021_wsb.csv', dtype=dtype_dictionary, 
                 keep_default_na=False, 
                 error_bad_lines=False,
                 na_values=['na',''],
                 usecols=['author','parent_author','created_utc'],chunksize=1000000)

df_comments = pd.concat(df_comments_chunked, ignore_index=True)
df_comments.created_utc = pd.to_datetime(df_comments.created_utc,unit='s')
print('Done reading!')

start = df_comments.created_utc.min()
period = df_comments.created_utc.max()-df_comments.created_utc.min()
step = dt.timedelta(weeks=week)

graph_dict = {}
if windowed:
    print('Windowed')
    for i in tqdm(range(int(np.ceil(period / step)))):
        before = start + step*i + step
        after = start + step*i
        df_comments_period = df_comments[(df_comments.created_utc < before) & (df_comments.created_utc > after)]

        G = nx.Graph()
        G.add_edges_from(df_comments_period[['author','parent_author']].dropna().values)

        folder_name = 'graphs_windowed_' + str(week) + '/'
        file_name = 'graph_' + str(after)[:10] + '_' + str(before)[:10] + '.gpickle'
        nx.write_gpickle(G, '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/' + folder_name + file_name)
    print('Saved to: ' + '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/' + folder_name + file_name)

else:
    print('Not windowed')
    for i in tqdm(range(int(np.ceil(period / step)))):
        before = start + step*i + step
        df_comments_before=df_comments.loc[df_comments.created_utc<before]
        
        G = nx.Graph()
        G.add_edges_from(df_comments_before[['author','parent_author']].dropna().values)

        folder_name = 'graphs_' + str(week) + '/'
        file_name = 'graph_' + str(before)[:10] + '.gpickle'
        nx.write_gpickle(G, '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/' + folder_name + file_name)
    print('Saved to: ' + '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/' + folder_name + file_name)

print('Done!')