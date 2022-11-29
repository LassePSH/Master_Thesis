import pandas as pd
import numpy as np
import praw
import datetime
from tqdm import tqdm
tqdm.pandas()

# Read-only instance
def get_reddit_instance():
    reddit = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                                client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                                user_agent="Scrapper")
    return reddit

def read_comments(path):
    df_comments=pd.read_csv(path)

    if 'author' not in df_comments.columns:
        df_comments.columns = ['author','text','created_utc','id','parent_id','score','subreddit','created']

    df_comments['date'] = pd.to_datetime(df_comments['created'],unit='s')
    df_comments['author'].dropna(inplace=True)
    df_comments = df_comments[df_comments['date'].apply(lambda x: isinstance(x, datetime.datetime))]
    return df_comments

def get_n_awards(id):
    comment_awards = []
    comment = reddit.comment(id=id)
    comment_awards.append(comment.all_awardings)
    # get number of awards
    return [len(x) for x in comment_awards][0]


subreddit = input("subreddit: ")

path = '/home/pelle/Master_Thesis/data/raw/'+subreddit+'_2015-01-01-2022-01-01/'
print('Reading data... '+'\n')
df=read_comments(path+subreddit+'_comments.csv')
print('Done reading data... '+'\n')

print('Connecting to Reddit... '+'\n')
reddit = get_reddit_instance()
print('Done connecting to Reddit... '+'\n')

print('Getting awards... '+'\n')
df['n_awards'] = df['id'].progress_apply(get_n_awards)

print('Saving data... '+'\n')
df[['id','n_awards']].to_csv(path+subreddit+'_awards_comments.csv',index=False)
