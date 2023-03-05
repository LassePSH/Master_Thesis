print('started..')

import pandas as pd
import numpy as np
import networkx as nx
import datetime as dt
import os
from tqdm import tqdm

week = int(input("Input period size (weeks): "))

### LOAD DATA ###
df_comments = pd.read_csv('comments_pmaw_2016-2021_wsb.csv',usecols=['author','parent_author','score','created_utc'])
df_comments = df_comments[df_comments['author'] != '[deleted]']
df_comments.created_utc = pd.to_datetime(df_comments.created_utc,unit='s')
print('Comments loaded')

df_post = pd.read_csv('submissions_pmaw_2016-2021_wsb.csv',usecols=['author','score','award_count','created_utc'])
df_post = df_post[df_post['author'] != '[deleted]']
df_post.created_utc = pd.to_datetime(df_post.created_utc)
print('Posts loaded')

### directory ###
folder_name = 'graphs_periode_' + str(week) + '/'
print('folder_name: ', folder_name+'\n')

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print('Created folder: ', folder_name)
else:
    print('Folder already exists: ', folder_name)
    print('Overwriting files in folder..')

### graph construction ###

start = df_comments.created_utc.min()
period = df_comments.created_utc.max()-df_comments.created_utc.min()
step = dt.timedelta(weeks=week)

print('start: ', start)
print('period: ', period)
print('step: ', step)

for i in tqdm(range(int(np.ceil(period / step)))):
    end = start + step*i + step
    begin = start + step*i
    df_comments_period = df_comments[(df_comments.created_utc < end) & (df_comments.created_utc > begin)]
    df_post_period = df_post[(df_post.created_utc < end) & (df_post.created_utc > begin)]

    G = nx.Graph()
    G.add_edges_from(df_comments_period[['author','parent_author']].dropna().values)

    file_name = 'graph_' + str(begin)[:10] + '_' + str(end)[:10] + '.gpickle'
    nx.write_gpickle(G, folder_name + file_name)

    ### graph analysis ###
    nodes = list(G.nodes)
    df_nodes = pd.DataFrame(nodes, columns=['author'])

    df_nodes = df_nodes.set_index('author')
    df_nodes = df_nodes.join(pd.concat([df_post_period[['author','score']],
            df_comments_period[['author','score']]]).groupby('author').mean()['score'].rename('mean_score'))

    df_nodes = df_nodes.join(pd.concat([df_post_period[['author','score']],
            df_comments_period[['author','score']]]).groupby('author').sum()['score'].rename('sum_score'))

    df_nodes = df_nodes.join(df_post_period[['author','award_count']].groupby('author').mean()['award_count'].rename('mean_awards'))
    df_nodes = df_nodes.join(df_post_period[['author','award_count']].groupby('author').sum()['award_count'].rename('sum_awards'))


    df_nodes=pd.DataFrame.from_dict(dict(G.degree()), orient='index', columns=['degree']).join(df_nodes)
    df_nodes=pd.DataFrame.from_dict(dict(nx.betweenness_centrality(G)), orient='index', columns=['Betweenness Centrality']).join(df_nodes)
    df_nodes=pd.DataFrame.from_dict(dict(nx.degree_centrality(G)), orient='index', columns=['Degree Centrality']).join(df_nodes)
    df_nodes=pd.DataFrame.from_dict(dict(nx.closeness_centrality(G)), orient='index', columns=['Closeness Centrality']).join(df_nodes)
    df_nodes=pd.DataFrame.from_dict(dict(nx.eigenvector_centrality(G)), orient='index', columns=['Eigenvector Centrality']).join(df_nodes)
    df_nodes=pd.DataFrame.from_dict(dict(nx.clustering(G)), orient='index', columns=['Clustering Coefficient']).join(df_nodes)
    df_nodes=pd.DataFrame.from_dict(dict(nx.average_neighbor_degree(G)), orient='index', columns=['Average Neighbor Degree']).join(df_nodes)

    ### SAVE ###
    df_nodes.to_csv(folder_name + 'nodes_' + str(begin)[:10] + '_' + str(end)[:10] + '.csv')

print('Done')