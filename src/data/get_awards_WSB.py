import pandas as pd 
import numpy as np
import os
import sys
from tqdm import tqdm
import praw

print("Reading data...")

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
                usecols=['id'])

print('Done loading data..' )

p = '/home/pelle/Master_Thesis/data/processed/wallstreetbets_scores/'
# create new dataframe if not already created
if not os.path.exists(p+'df_awards_post.csv'):
    print('Creating new dataframe')
    df_id=pd.DataFrame()
    df_id.to_csv(p+'df_awards_post.csv',index=False)
    len_id = 0
else: 
    df_id=pd.read_csv(p+'df_awards_post.csv')
    len_id=len(df_id)
    print('starting from: ',len_id)
    df_id = None



# Read-only instance
def get_reddit_instance():
    reddit = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                                client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                                user_agent="Scrapper")
    return reddit

reddit = get_reddit_instance()

def get_n_awards(id):
    submission_awards = []
    submission = reddit.submission(id=id)
    submission_awards.append(submission.all_awardings)
    # get number of awards
    return [len(x) for x in submission_awards]

print('Lets go!!!')

# Get the awards for a post and append to the dataframe. skip len_id rows
for id in tqdm(df_posts['id'][len_id:]):
    # append to csv file
    N_awards = get_n_awards(id)
    df_awards = pd.DataFrame({'id':id,'N_awards':N_awards})
    df_awards.to_csv(p+'df_awards_post.csv',mode='a',header=False,index=False)



