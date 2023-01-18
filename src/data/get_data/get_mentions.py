import pandas as pd
import numpy as np

### data ###
p = '/home/pelle/Master_Thesis/data/raw/wallstreetbets/'
df_post = pd.read_csv(p+'submissions_pmaw_2016-2021_wsb.csv',usecols=['author','created_utc','score','num_comments','title','selftext','id','award_count'])
df_post = df_post[df_post['author'] != '[deleted]']
df_post['subreddit']='wallstreetbets'
df_post.rename(columns={'created_utc':'date','num_comments':'n_comments','selftext':'text','id':'id','award_count':'n_awards'},inplace=True)
df_post = df_post[['author','date','score','n_comments','title','text','id','n_awards','subreddit']]
print('Posts loaded')

df_comments = pd.read_csv(p+'comments_pmaw_2016-2021_wsb.csv',usecols=['author','parent_author','created_utc','body'])
df_comments = df_comments[df_comments['author'] != '[deleted]']
df_comments.created_utc = pd.to_datetime(df_comments.created_utc,unit='s')
print('Comments loaded')

df_post_balanced = pd.read_csv('/home/pelle/Master_Thesis/src/Heisenberg_SSH/balanced_data_chunked10.csv')
df_post_balanced['date'] = pd.to_datetime(df_post_balanced['date'])

def get_mentions_in_period(pre_date,date,author):
    df_comments_sub = df_comments[(df_comments['created_utc'] >= pre_date) & (df_comments['created_utc'] < date)]
    df_post_sub = df_post_balanced[(df_post_balanced['date'] >= pre_date) & (df_post_balanced['date'] < date)]


    user_name = '/u/'+author
    mentions = df_post_sub.text.str.contains(user_name).sum() + df_post_sub.title.str.contains(user_name).sum() + df_comments_sub.body.str.contains(user_name).sum()

    return mentions

if __name__ == '__main__':
    df_post_balanced['mentions'] = df_post_balanced.apply(lambda row: get_mentions_in_period(row['pre_date'],row['date'],row['author']),axis=1)
    df_post_balanced.to_csv('/home/pelle/Master_Thesis/src/Heisenberg_SSH/balanced_data_chunked10.csv')

