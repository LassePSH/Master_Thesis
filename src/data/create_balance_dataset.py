import pandas as pd
import numpy as np
import networkx as nx
import datetime
import os

N_c = int(input('Chunk number: '))

p = '/home/pelle/Master_Thesis/data/raw/wallstreetbets/'
df_post = pd.read_csv(p+'submissions_pmaw_2016-2021_wsb.csv',usecols=['author','created_utc','score','num_comments','title','selftext','id','award_count'])
df_post = df_post[df_post['author'] != '[deleted]']
df_post = df_post[df_post['author'] != 'AutoModerator']
df_post = df_post.dropna(subset=['author'])

# rename columns
df_post.rename(columns={'created_utc':'date','num_comments':'n_comments','selftext':'text','id':'id','award_count':'n_awards'},inplace=True)
df_post = df_post[['author','date','score','n_comments','title','text','id','n_awards']]
print('Posts loaded')

len1=len(df_post.loc[df_post['n_awards'] > 0])
len2=len(df_post.loc[df_post['n_awards'] == 0])

print('N post with awards: ' + str(len1))
print('N post with no awards: ' + str(len2))
print()

df_post_awarded = df_post.loc[df_post['n_awards'] > 0]
df_post_not_awarded = df_post.loc[df_post['n_awards'] == 0]
df_post_awarded['date'] = pd.to_datetime(df_post_awarded['date'])
df_post_not_awarded['date'] = pd.to_datetime(df_post_not_awarded['date'])


df_post_not_awarded['date_no_time'] = df_post_not_awarded['date'].dt.date
df_post_awarded['date_no_time'] = df_post_awarded['date'].dt.date

df_sampled = pd.DataFrame()
for i in (df_post_awarded['date_no_time'].unique()):
    df_temp = df_post_not_awarded[df_post_not_awarded['date_no_time'] == i]
    df_sampled = pd.concat([df_sampled, df_temp.sample(n=len(df_post_awarded[df_post_awarded['date_no_time'] == i]), random_state=1)])

# drop duplicates
df_sampled = df_sampled.drop_duplicates(subset=['id'])
df_post_balanced = pd.concat([df_post_awarded, df_sampled])

print('Balanced!')
print('N post with awards: '+str(len(df_post_awarded)))
print('N post with no awards: '+str(len(df_sampled)))

# Cleaning
df_post_balanced['date'] = pd.to_datetime(df_post_balanced['date'])
df_post_balanced['text'].fillna('', inplace=True)
df_post_balanced['title'].fillna('', inplace=True)
df_post_balanced['text'] = df_post_balanced['text'].astype(str)
df_post_balanced['title'] = df_post_balanced['title'].astype(str)
df_post_balanced['author'] = df_post_balanced['author'].astype(str)
df_post_balanced['id'] = df_post_balanced['id'].astype(str)

for char in ['\n','\r','\t']:
    df_post_balanced['text'] = df_post_balanced['text'].str.replace(char, ' ')
    df_post_balanced['title'] = df_post_balanced['title'].str.replace(char, ' ')
df_post_balanced['text_title'] = df_post_balanced['title'] + ' ' + df_post_balanced['text']

# getting predate!
df_post_balanced['pre_date'] = df_post_balanced['date']-pd.Timedelta(weeks=10) 

#makeing chunks
df_post_balanced['chunk'] = np.arange(len(df_post_balanced)) // (len(df_post_balanced)/N_c)

#saving
df_post_balanced.to_csv('true_balanced_data_chunked'+str(N_c)+'.csv',index=False)
print('Saved!')