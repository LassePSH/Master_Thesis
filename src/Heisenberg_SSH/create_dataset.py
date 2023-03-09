import pandas as pd
import numpy as np
import networkx as nx
import datetime
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# choose chunk 0,1,2,3
N_c = int(input('Chunk number: '))

### data ###
df_post = pd.read_csv('submissions_pmaw_2016-2021_wsb.csv',usecols=['author','created_utc','score','num_comments','title','selftext','id','award_count'])
df_post=df_post[df_post.author!='[deleted]']
df_post=df_post[df_post.author!='AutoModerator']
df_post=df_post.dropna(subset=['author'])

df_post['subreddit']='wallstreetbets'
df_post.rename(columns={'created_utc':'date','num_comments':'n_comments','selftext':'text','id':'id','award_count':'n_awards'},inplace=True)
df_post = df_post[['author','date','score','n_comments','title','text','id','n_awards','subreddit']]
df_post['awarded'] = df_post['n_awards'].apply(lambda x: 1 if x>0 else 0)
print('Posts loaded')

df_comments = pd.read_csv('comments_pmaw_2016-2021_wsb.csv',usecols=['author','parent_author','created_utc','body'])
df_comments=df_comments[df_comments.author!='[deleted]']
df_comments=df_comments[df_comments.author!='AutoModerator']
df_comments=df_comments.dropna(subset=['author','parent_author'])

df_comments.created_utc = pd.to_datetime(df_comments.created_utc,unit='s')
print('Comments loaded')

df_post_balanced = pd.read_csv('true_balanced_data_chunked10.csv')
df_post_balanced = df_post_balanced[df_post_balanced['author'] != '[deleted]']
df_post_balanced=df_post_balanced[df_post_balanced.author!='AutoModerator']
print('true_balanced_data_chunked10.csv')
print('Balanced loaded')

def degree_centrality(degree,N):
    return degree/(N-1)

def get_graph_WSB(pre_date,date,author,id):
    df_comments_sub = df_comments[(df_comments['created_utc'] >= pre_date) & (df_comments['created_utc'] < date)]
    df_comments_sub = df_comments_sub.dropna(subset=['author','parent_author'])
    df_post_sub = df_post_balanced[(df_post_balanced['date'] >= pre_date) & (df_post_balanced['date'] < date)]

    # create graph
    G = nx.from_pandas_edgelist(df_comments_sub, 'author', 'parent_author', create_using=nx.Graph())
    G_di = nx.from_pandas_edgelist(df_comments_sub, 'author', 'parent_author', create_using=nx.DiGraph())

    # directed graph
    N_nodes = len(G.nodes)
    N_edges = G.number_of_edges()

    # reciprocal edges for directed graph
    N_rec=len([e for e in G_di.edges if G_di.has_edge(e[1], e[0])])
    frac_rec = N_rec/N_edges

    # previus awarded
    prev_N_awarded = df_post.loc[df_post['author']==author].loc[df_post['date'] < date].awarded.sum()

    if author in df_comments_sub['author'].values or author in df_comments_sub['parent_author'].values:
        # undirected graph
        degree = G.degree(author)
        degree_cen = degree_centrality(degree,N_nodes)
        close_cen  = nx.closeness_centrality(G, u=author)  
        
        # directed graph
        degree_in = G_di.in_degree(author)
        degree_out = G_di.out_degree(author)
        N_rec_author = len([e for e in G_di.edges if G_di.has_edge(e[1], e[0]) and e[0]==author])


    else:
        degree_cen = 0
        close_cen = 0
        degree = 0
        degree_in = 0
        degree_out = 0
        N_rec_author = 0

    user_name = '/u/'+author
    mentions = df_post_sub.text.str.contains(user_name).sum() + df_post_sub.title.str.contains(user_name).sum() + df_comments_sub.body.str.contains(user_name).sum()
    user = '/u/'
    sum_mentions = df_post_sub.text.str.contains(user).sum() + df_post_sub.title.str.contains(user).sum() + df_comments_sub.body.str.contains(user).sum()
    
    activity = len(df_comments_sub.loc[df_comments_sub['author']==author]) + len(df_post_sub.loc[df_post_sub['author']==author])

    return [degree_cen, close_cen,activity,degree, N_nodes, N_edges,mentions,sum_mentions,id,frac_rec,N_rec,degree_in,degree_out,N_rec_author,prev_N_awarded]

if __name__ == '__main__':
    
    # choose a subset of the data
    print('Chunk size of data: ' + str(len(df_post_balanced.loc[df_post_balanced['chunk'] == N_c])/len(df_post_balanced)))
    df_post_balanced = df_post_balanced.loc[df_post_balanced['chunk'] == N_c]

    col_names = ['degree_cen', 'close_cen',
                 'activity','degree', 
                 'N_nodes', 'N_edges',
                 'mentions','sum_mentions',
                 'id','frac_rec',
                 'N_rec','degree_in',
                 'degree_out','N_rec_author',
                 'prev_N_awarded']

    print('Computing graph features..')

    pre_date = df_post_balanced['pre_date'].values
    date = df_post_balanced['date'].values
    author = df_post_balanced['author'].values
    id = df_post_balanced['id'].values


    file_name = 'true_graph_features_chunk_' + str(N_c) + '.csv'
    df_out=pd.DataFrame(columns=col_names)
    df_out.to_csv(file_name,index=False,header=True)
    
    print('Data size: ' + str(len(id)))
    
    for i,j,k,l in tqdm(zip(pre_date,date,author,id)):
        out = get_graph_WSB(i,j,k,l)
        # save to the csv
        df_out = pd.DataFrame([out],
        columns=col_names,index=[0])
        df_out.to_csv(file_name,index=False,mode='a',header=False)


print('Done!')



