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

def read_posts(path):
    df=pd.read_csv(path)
    df.columns = ['author','created_utc','domain','id','n_comments','score','text','title','url','date']
    df['date'] = pd.to_datetime(df['date'],unit='s')
    # drop floats in date column
    df = df[df['date'].apply(lambda x: isinstance(x, datetime.datetime))]
    df['author'].dropna(inplace=True)
    df.drop(df.loc[df['author']=='[deleted]'].index, inplace=True)
    return df

def get_n_awards(id):
    submission_awards = []
    submission = reddit.submission(id=id)
    submission_awards.append(submission.all_awardings)
    y = 0
    for a in submission_awards[0]:
        y = y + a['count']
    return y


subreddit = input("subreddit: ")

path = '/home/pelle/Master_Thesis/data/raw/'+subreddit+'_2015-01-01-2022-01-01/'
print('Reading data... '+'\n')
df=read_posts(path+subreddit+'_posts.csv')
print('Done reading data... '+'\n')

print('Connecting to Reddit... '+'\n')
reddit = get_reddit_instance()
print('Done connecting to Reddit... '+'\n')

print('Getting awards... '+'\n')
df['n_awards'] = df['id'].progress_apply(get_n_awards)

print('Saving data... '+'\n')
df[['id','n_awards']].to_csv("/home/pelle/Master_Thesis/awards/"+subreddit+'_awards.csv',index=False)
