import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

folder_name = input("Folder name: ")
get_scores = input("Get scores? (y/n): ")
get_scores = get_scores == "n"
get_graph_analysis = input("Get graphs? (y/n): ")
get_graph_analysis = get_graph_analysis == "y"

files = os.listdir('/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+folder_name)
files = [file for file in files if file[-8:] == '.gpickle']
print("Number of .gpickle files : ", len(files))

##################### Graph Analysis #####################

if get_graph_analysis: 
    df_path = '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/' + folder_name +'/df_final.csv'
    print('Saving to: ', df_path)
    dict = {}

    for i in tqdm(files):
        with open('/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+folder_name+'/' +i, 'rb') as handle:
            G = pickle.load(handle)
            N_nodes=len(G.nodes)
            N_edges=len(G.edges)
            Mean_degree=np.mean([G.degree[n] for n in G.nodes]) 
            clustering_coefficient=nx.average_clustering(G)
            date = i[6:16]
            dict[date] = [N_nodes, N_edges, Mean_degree, clustering_coefficient]

    df_final = pd.DataFrame.from_dict(dict, orient='index', columns=['N_nodes', 'N_edges', 'mean_degree', 'clustering_coefficient'])
    df_final.to_csv(df_path)

    # df_final = pd.read_csv(df_path)
    # df_final['date'] = pd.to_datetime(df_final['Unnamed: 0'])
    # df_final = df_final.set_index('date')
    # df_final.sort_index(inplace=True)
    # df_final = df_final.drop(columns=['Unnamed: 0'])

    print("Done with graph analysis! " +'\n')

##################### getting scores #####################

if get_scores: exit()

print('Reading data... '+'\n')

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

# make new folder
if not os.path.exists('/home/pelle/Master_Thesis/data/processed/wallstreetbets_scores/'+folder_name):
    os.makedirs('/home/pelle/Master_Thesis/data/processed/wallstreetbets_scores/'+folder_name)

for file in tqdm(files):
    before = pd.to_datetime(file[-18:-8])

    with open('/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+folder_name+'/' +file, 'rb') as handle:
        G = pickle.load(handle)

    # degree_centrality = nx.degree_centrality(G)
    # betweenness_centrality = nx.betweenness_centrality(G)
    clustering_coefficient = nx.clustering(G)
    degree = nx.degree(G)

    df_comments_before=df_comments.loc[df_comments.created_utc<before]   
    df_posts_before=df_posts.loc[df_posts.created_utc<before]

    df_authors_period=pd.concat([df_posts_before[['author','score']],df_comments_before[['author','score']]])
    s=df_authors_period.groupby('author').sum().rename(columns={'score':'sum_score'})
    m=df_authors_period.groupby('author').mean().rename(columns={'score':'mean_score'})
    n=df_authors_period.groupby('author').count().rename(columns={'score':'activity'})
    df_score_before=s.join(m).join(n)
    df_score_before['date']=before

    # df_score_before['degree_centrality'] = df_score_before.index.map(degree_centrality)
    # df_score_before['betweenness_centrality'] = df_score_before.index.map(betweenness_centrality)
    df_score_before['degree'] = df_score_before.index.map(degree)
    df_score_before['clustering_coefficient'] = df_score_before.index.map(clustering_coefficient)

    df_score_before.to_csv('/home/pelle/Master_Thesis/data/processed/wallstreetbets_scores/'+folder_name+'/df_score_before_'+str(before.date())+'.csv')

print('Done!!')