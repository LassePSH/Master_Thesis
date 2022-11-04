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
get_scores = input("Input get_scores (y/n): ")
get_scores = get_scores == "y"

#################### LOAD DATA ############################

print('Reading data...')

sample = pd.read_csv('/home/pelle/Downloads/comments_pmaw_2016-2021_wsb.csv',nrows=10)
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


df_comments_chunked = pd.read_csv('/home/pelle/Downloads/comments_pmaw_2016-2021_wsb.csv', dtype=dtype_dictionary, 
                 keep_default_na=False, 
                #  error_bad_lines=False,
                 on_bad_lines='warn',
                 na_values=['na',''],
                 usecols=['author','parent_author','created_utc','score'],chunksize=1000000)
df_comments = pd.concat(df_comments_chunked, ignore_index=True)
df_comments.created_utc = pd.to_datetime(df_comments.created_utc,unit='s')

if get_scores:
    sample = pd.read_csv('/home/pelle/Downloads/submissions_pmaw_2016-2021_wsb.csv',nrows=10)
    dtypes = sample.dtypes # Get the dtypes
    cols = sample.columns # Get the columns
    dtype_dictionary = {} 
    for c in cols:
        if str(dtypes[c]) == 'int64':
            dtype_dictionary[c] = 'float32' # Handle NANs in int columns
        else:
            dtype_dictionary[c] = str(dtypes[c])

    df_posts = pd.read_csv('/home/pelle/Downloads/submissions_pmaw_2016-2021_wsb.csv',dtype=dtype_dictionary, 
                    keep_default_na=False,
                    na_values=['na',''],
                    usecols=['author','created_utc','score'])
    df_posts.created_utc = pd.to_datetime(df_posts.created_utc,unit='s')

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
step = dt.timedelta(weeks=week)

if windowed:
    print('Windowed')
    for i in tqdm(range(int(np.ceil(period / step)))):
        before = start + step*i + step
        after = start + step*i
        df_comments_period = df_comments[(df_comments.created_utc < before) & (df_comments.created_utc > after)]
        if get_scores: 
            df_posts_period = df_posts[(df_posts.created_utc < before) & (df_posts.created_utc > after)]
            df_authors_period=pd.concat([df_posts_period[['author','score']],df_comments_period[['author','score']]])
            s=df_authors_period.groupby('author').sum().rename(columns={'score':'sum_score'})
            m=df_authors_period.groupby('author').mean().rename(columns={'score':'mean_score'})
            df_score_period=s.join(m)

        G = nx.Graph()
        G.add_edges_from(df_comments_period[['author','parent_author']].dropna().values)

        # add attributes to nodes
        if get_scores: 
            for node in G.nodes():  
                if node in df_score_period.index:
                    G.nodes[node]['sum_score'] = df_score_period.loc[node,'sum_score']
                    G.nodes[node]['mean_score'] = df_score_period.loc[node,'mean_score']
                else:
                    print(node + ' not in df_score_period')
                    # G.nodes[node]['sum_score'] = 0
                    # G.nodes[node]['mean_score'] = 0

        file_name = 'graph_' + str(after)[:10] + '_' + str(before)[:10] + '.gpickle'
        nx.write_gpickle(G, folder_name + file_name)

else:
    print('Not windowed')
    for i in tqdm(range(int(np.ceil(period / step)))):
        before = start + step*i + step
        df_comments_before=df_comments.loc[df_comments.created_utc<before]
        if get_scores: 
            df_posts_before=df_posts.loc[df_posts.created_utc<before]
            df_authors_period=pd.concat([df_posts_before[['author','score']],df_comments_before[['author','score']]])
            s=df_authors_period.groupby('author').sum().rename(columns={'score':'sum_score'})
            m=df_authors_period.groupby('author').mean().rename(columns={'score':'mean_score'})
            n=df_authors_period.groupby('author').count().rename(columns={'score':'count'})
            df_score_before=s.join(m)
        
        G = nx.Graph()
        G.add_edges_from(df_comments_before[['author','parent_author']].dropna().values)

        # add attributes to nodes
        if get_scores:
            for node in G.nodes():  
                G.nodes[node]['sum_score'] = df_score_before.loc[node].sum_score
                G.nodes[node]['mean_score'] = df_score_before.loc[node].mean_score
            else:
                print(node + ' not in df_score_period')

        file_name = 'graph_' + str(before)[:10] + '.gpickle'
        nx.write_gpickle(G, folder_name + file_name)

print('Done!')

# to do:
# - add weights to edges
# - add directed edges