import pandas as pd
import networkx as nx

def read_posts(path):
    df=pd.read_csv(path)
    df.columns = ['author','created_utc','domain','id','n_comments','score','text','title','url','date']
    df['date'] = pd.to_datetime(df['date'],unit='s')
    df['author'].dropna(inplace=True)
    df.drop(df.loc[df['author']=='[deleted]'].index, inplace=True)
    return df


def read_comments(path):
    df_comments=pd.read_csv(path)
    df_comments['date'] = pd.to_datetime(df_comments['created'],unit='s')
    df_comments['author'].dropna(inplace=True)
    return df_comments


def get_all_nodes(df,df_comments):
    df_all_nodes = pd.DataFrame()
    df_all_nodes['author'] = pd.concat([df_comments["author"].drop_duplicates(), df["author"].drop_duplicates()])
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes['author'].dropna(inplace=True)

    def find_type(author):
        if author in df["author"].unique():
            if author in df_comments["author"].unique():
                return "both"
            else:
                return "poster"
        else:
            return "commenter"

    df_all_nodes["type"] = df_all_nodes["author"].apply(find_type)

    #remove float values from df_all_nodes
    df_all_nodes = df_all_nodes[df_all_nodes["author"].apply(lambda x: isinstance(x, str))]
    
    return df_all_nodes


def get_graph(df,df_comments,df_all_nodes):
    G = nx.Graph()

    df_all_nodes["author"].dropna(inplace=True)
    G.add_nodes_from(df_all_nodes["author"].loc[df_all_nodes['type']=='both'], type='both')
    G.add_nodes_from(df_all_nodes["author"].loc[df_all_nodes['type']=='commenter'], type='commenter')
    G.add_nodes_from(df_all_nodes["author"].loc[df_all_nodes['type']=='poster'], type='poster')

    color_map = []
    for node in G:
        if type(node) == float: print('node is float!')

        if G.nodes[node]['type'] == 'both': color_map.append('green')
        elif G.nodes[node]['type'] == 'commenter': color_map.append('red')
        else: color_map.append('blue')

        for p_id in df_comments.loc[df_comments['author'] == node].parent_id:
            if len(list(p_id)) > 0:
                if 't3_' in p_id:
                    p_id = p_id.replace('t3_','')
                    if p_id in df['id'].unique():
                        if df.loc[df['id'] == p_id].author.values[0] in G.nodes:

                            if G.has_edge(node, df.loc[df['id'] == p_id].author.values[0]): w = w + 1
                            else: w = 1
                            
                            G.add_edge(node, df.loc[df['id'] == p_id].author.values[0], weight=w)
                
                elif 't1_' in p_id:
                    p_id = p_id.replace('t1_','')
                    if p_id in df_comments['id'].unique():
                        if df_comments.loc[df_comments['id'] == p_id].author.values[0] in G.nodes:

                            if G.has_edge(node, df_comments.loc[df_comments['id'] == p_id].author.values[0]): w = w + 1
                            else: w = 1

                            G.add_edge(node, df_comments.loc[df_comments['id'] == p_id].author.values[0], weight=w)
    
    
    return G


def get_DiGraph(df,df_comments,df_all_nodes):
    G_di = nx.DiGraph()

    df_all_nodes["author"].dropna(inplace=True)
    G_di.add_nodes_from(df_all_nodes["author"].loc[df_all_nodes['type']=='both'], type='both')
    G_di.add_nodes_from(df_all_nodes["author"].loc[df_all_nodes['type']=='commenter'], type='commenter')
    G_di.add_nodes_from(df_all_nodes["author"].loc[df_all_nodes['type']=='poster'], type='poster')

    color_map = []
    for node in G_di:
        if type(node) == float: print(node)


        if G_di.nodes[node]['type'] == 'both':color_map.append('green')
        elif G_di.nodes[node]['type'] == 'commenter':color_map.append('red')
        else: color_map.append('blue')

        for p_id in df_comments.loc[df_comments['author'] == node].parent_id:
            if len(list(p_id)) > 0:

                if 't3_' in p_id:
                    p_id = p_id.replace('t3_','')
                    if p_id in df['id'].unique():
                        if df.loc[df['id'] == p_id].author.values[0] in G_di.nodes:

                            if G_di.has_edge(node, df.loc[df['id'] == p_id].author.values[0]): w = w + 1
                            else: w = 1
                            
                            G_di.add_edge(node, df.loc[df['id'] == p_id].author.values[0], weight=w)
                
                elif 't1_' in p_id:
                    p_id = p_id.replace('t1_','')
                    if p_id in df_comments['id'].unique():
                        if df_comments.loc[df_comments['id'] == p_id].author.values[0] in G_di.nodes:

                            if G_di.has_edge(node, df_comments.loc[df_comments['id'] == p_id].author.values[0]): w = w + 1
                            else: w = 1

                            G_di.add_edge(node, df_comments.loc[df_comments['id'] == p_id].author.values[0], weight=w)

    return G_di


def get_biggest_component(G):
    G2 = G.copy()
    G2.remove_edges_from(nx.selfloop_edges(G2))
    G2 = [G2.subgraph(cc) for cc in nx.connected_components(G2)][0]

    return G2


def get_comment_post_date(df_comments,df):
    df_comment_post=pd.concat([df_comments[['date','author']],df[['date','author']]]).sort_values(by='date')
    df_comment_post.dropna(inplace=True) 

    def max_date(author):
        df_a=df_comment_post.loc[df_comment_post['author']==author]
        return df_a['date'].max()

    def min_date(author):
        df_a=df_comment_post.loc[df_comment_post['author']==author]
        return df_a['date'].min()

    df_comment_post['min_date'] = df_comment_post['author'].apply(lambda x: min_date(x))
    df_comment_post['max_date'] = df_comment_post['author'].apply(lambda x: max_date(x))
    df_comment_post['delta_time'] = df_comment_post['max_date'] - df_comment_post['min_date']

    return df_comment_post


def cluster_coefficient_swapped_pvalue(G):
    average_clustering = []
    for i in range(1000):
        G_swapped = G.copy()
        G_swapped = nx.double_edge_swap(G_swapped, nswap=len(G.nodes), max_tries=len(G.nodes)*2)
        average_clustering.append(nx.average_clustering(G_swapped))
    
    p_value = np.array(np.array(average_clustering) > nx.average_clustering(G)).sum() / len(average_clustering)
    
    print('Clustering coefficient p-value: ', p_value)


def describe_graph(G):
    print('Number of nodes: ', len(G.nodes))
    print('Number of edges: ', len(G.edges))
    print('Average weight of edges: ', np.mean([G.edges[e]['weight'] for e in G.edges]))
    print('Average degree: ', np.mean([G.degree[n] for n in G.nodes]))


def reciprocal_edges(G_di):
    print('Number of reciprocal edges: ', len([e for e in G_di.edges if G_di.has_edge(e[1], e[0])]))
    print('Fraction of reciprocal edges: ', len([e for e in G_di.edges if G_di.has_edge(e[1], e[0])])/len(G_di.edges))


def get_authors(G,df_all_nodes,df_comments,df,df_comment_post):
    df_authors=df_all_nodes.set_index('author')
    # df_authors = df_authors.join(df_karma.set_index('author'))
    df_authors = df_authors.join(pd.concat([df[['author','score']],df_comments[['author','score']]]).groupby('author').mean()['score'].rename('sum_score'))
    df_authors = df_authors.join(pd.concat([df[['author','score']],df_comments[['author','score']]]).groupby('author').sum()['score'].rename('mean_score'))

    # df_authors = df_authors.join(pd.DataFrame(df_comments.groupby('author')['score'].sum()).rename('sum_score_comments'))
    # df_authors = df_authors.join(pd.DataFrame(df_comments.groupby('author')['score'].mean().rename('mean_score_comment')))


    df_authors=pd.DataFrame.from_dict(dict(G.degree()), orient='index', columns=['degree']).join(df_authors)
    df_authors=pd.DataFrame.from_dict(dict(nx.betweenness_centrality(G)), orient='index', columns=['Betweenness Centrality']).join(df_authors)
    df_authors=pd.DataFrame.from_dict(dict(nx.degree_centrality(G)), orient='index', columns=['Degree Centrality']).join(df_authors)
    df_authors=pd.DataFrame.from_dict(dict(df_comment_post["author"].value_counts()), orient='index', columns=['Activity']).join(df_authors)

    return df_authors