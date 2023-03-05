print('started..')

import pandas as pd
import numpy as np
import networkx as nx
import datetime

print('Starting: ', datetime.datetime.now())

### LOAD DATA ###
df_comments = pd.read_csv('comments_pmaw_2016-2021_wsb.csv',usecols=['author','parent_author','score'])
df_comments = df_comments[df_comments['author'] != '[deleted]']
print('Comments loaded')

df_post = pd.read_csv('submissions_pmaw_2016-2021_wsb.csv',usecols=['author','score','award_count'])
df_post = df_post[df_post['author'] != '[deleted]']
print('Posts loaded')

### GRAPH ###
G = nx.Graph()
G.add_edges_from(df_comments[['author','parent_author']].dropna().values)
print('Graph constructed')

### METRICS ###
nodes = list(G.nodes)
df_nodes = pd.DataFrame(nodes, columns=['author'])
print('Empty dataframe constructed')

df_nodes = df_nodes.set_index('author')
df_nodes = df_nodes.join(pd.concat([df_post[['author','score']],df_comments[['author','score']]]).groupby('author').mean()['score'].rename('mean_score'))
print('Mean score done: ', datetime.datetime.now())
df_nodes = df_nodes.join(pd.concat([df_post[['author','score']],df_comments[['author','score']]]).groupby('author').sum()['score'].rename('sum_score'))
print('Sum score done: ', datetime.datetime.now())
df_nodes = df_nodes.join(df_post[['author','award_count']].groupby('author').mean()['award_count'].rename('mean_awards'))
print('Mean awards done: ', datetime.datetime.now())
df_nodes = df_nodes.join(df_post[['author','award_count']].groupby('author').sum()['award_count'].rename('sum_awards'))
print('Sum awards done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(G.degree()), orient='index', columns=['degree']).join(df_nodes)
print('Degree done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(nx.betweenness_centrality(G)), orient='index', columns=['Betweenness Centrality']).join(df_nodes)
print('Betweenness done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(nx.degree_centrality(G)), orient='index', columns=['Degree Centrality']).join(df_nodes)
print('Degree Centrality done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(nx.closeness_centrality(G)), orient='index', columns=['Closeness Centrality']).join(df_nodes)
print('Closeness done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(nx.eigenvector_centrality(G)), orient='index', columns=['Eigenvector Centrality']).join(df_nodes)
print('Eigenvector done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(nx.clustering(G)), orient='index', columns=['Clustering Coefficient']).join(df_nodes)
print('Clustering done: ', datetime.datetime.now())
df_nodes=pd.DataFrame.from_dict(dict(nx.average_neighbor_degree(G)), orient='index', columns=['Average Neighbor Degree']).join(df_nodes)
print('Average Neighbor Degree done: ', datetime.datetime.now())
print('Metrics calculated: ', datetime.datetime.now())

### SAVE ###
df_nodes.to_csv('wsb_metrics.csv')
print('Saved: ', datetime.datetime.now())