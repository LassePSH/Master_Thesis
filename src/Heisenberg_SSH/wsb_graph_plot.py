import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

sample = pd.read_csv('comments_pmaw_2016-2021_wsb.csv',nrows=10)
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

df_comments_chunked = pd.read_csv('comments_pmaw_2016-2021_wsb.csv', dtype=dtype_dictionary, 
    on_bad_lines='skip',
    chunksize=1000000,
    low_memory=False,
    usecols=['author','parent_author']
    )

df_comments = pd.concat(df_comments_chunked, ignore_index=True)

print('Done reading!'+'\n')

G = nx.Graph()
G.add_edges_from(df_comments[['author','parent_author']].dropna().values)
G.remove_edges_from(nx.selfloop_edges(G))
G = [G.subgraph(cc) for cc in nx.connected_components(G)][0]
print('Done creating graph!'+'\n')


# plot graph
print('Plotting graph...'+'\n')
plt.figure(figsize=(22,22))
s = len(G)*-0.03595226676061454+89.65811895967292+10
nx.draw(G, 
        node_color=c, 
        with_labels=False, 
        pos=nx.spring_layout(G, k=0.15, iterations=20), 
        node_size=s,
        width=0.04,
        alpha=0.6
        )

plt.savefig('wsb_graph.png', dpi=300, bbox_inches='tight')

print('Done plotting!'+'\n')
