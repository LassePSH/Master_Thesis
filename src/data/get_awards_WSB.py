import pandas as pd 
import numpy as np
import os
import sys
from tqdm import tqdm
import praw
import random
import string

# press enter to start
input("Press Enter to start...")
scrapper = input("scrapper1 (y/n): ")
scrapper = scrapper == "y"
hpc = input("hpc (y/n): ")
hpc = hpc == "y"

if scrapper:
    print('Using Scrapper')
else: 
    print('Using Scrapper2')

n_chunks = int(input("N chunks?: "))
print('Running ',n_chunks,' times')
# choose which chunk to start on
start_chunk = int(input("Start chunk?: ")) - 1
print('Starting on chunk ',start_chunk)


print("Reading data...")
if hpc:
    sample = pd.read_csv('/home/hpc/Master_Thesis/data/raw/wallstreetbets/submissions_pmaw_2016-2021_wsb.csv',nrows=10)
else:
    sample = pd.read_csv('/home/pelle/Master_Thesis/data/raw/wallstreetbets/submissions_pmaw_2016-2021_wsb.csv',nrows=10)
dtypes = sample.dtypes # Get the dtypes
cols = sample.columns # Get the columns
dtype_dictionary = {} 
for c in cols:
    if str(dtypes[c]) == 'int64':
        dtype_dictionary[c] = 'float32' # Handle NANs in int columns
    else:
        dtype_dictionary[c] = str(dtypes[c])

if hpc:
    df_posts = pd.read_csv('/home/hpc/Master_Thesis/data/raw/wallstreetbets/submissions_pmaw_2016-2021_wsb.csv',dtype=dtype_dictionary, 
                keep_default_na=False,
                na_values=['na',''],
                usecols=['id'])
else:
    df_posts = pd.read_csv('/home/pelle/Master_Thesis/data/raw/wallstreetbets/submissions_pmaw_2016-2021_wsb.csv',dtype=dtype_dictionary, 
                    keep_default_na=False,
                    na_values=['na',''],
                    usecols=['id'])

print('Done loading data..' )

print('Getting current awards from dir...')
p = '/home/pelle/Master_Thesis/awards/wallstreetbets/'
files = os.listdir(p)
print('Found ',len(files),' files')
print(files)
df_awards = pd.DataFrame()
for f in files:
    df = pd.read_csv(p+f,on_bad_lines='skip',header=None, names=['id','award_count'])
    df_awards = pd.concat([df_awards,df],ignore_index=True)
df_awards = df_awards.drop_duplicates(subset=['id'])

l = len(df_awards)
print('Found ',l,' awards')
l2 = len(df_posts)
df_posts=df_posts.loc[~df_posts['id'].isin(df_awards['id'])]
print('Missing ',(l2-len(df_posts))/l2*100,'% of posts')

def get_random_name():
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    return name

file_name=get_random_name()
file_name = file_name+'.csv'
print('Saving to: ',file_name)

p = '/home/pelle/Master_Thesis/awards/wallstreetbets/'
# create new dataframe if not already created
if not os.path.exists(p+file_name):
    print('Creating new dataframe')
    df_id=pd.DataFrame()
    df_id.to_csv(p+file_name,index=False)


# divide into chunks
df_posts = df_posts.sort_values('id')
df_posts = df_posts.reset_index(drop=True)
df_posts['chunk'] = df_posts.index // (len(df_posts)//n_chunks)
df_posts = df_posts.sort_values('chunk')


# Read-only instance
def get_reddit_instance(reverse):
    if scrapper:
        reddit = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                                    client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                                    user_agent="Scrapper")
    else: 
        reddit = praw.Reddit(client_id="FZ7X2_1D0FiZr1Gm6fYSoA",
                            client_secret="cYhLkIKcGgVtt26nhXhSOQwu55Lqdw",
                            user_agent="Scrapper2")
    return reddit


def get_n_awards(id):
    try:
        submission_awards = []
        submission = reddit.submission(id=id)
        submission_awards.append(submission.all_awardings)
        y = 0
        for a in submission_awards[0]:
            y = y + a['count']
    except:
        y = 0
        print('Error with id: ',id)
    return y


reddit = get_reddit_instance(scrapper)
print('Reddit instance created')
print('Lets go!!!')

# Get the awards for a post and append to the dataframe
df_chunk = df_posts.loc[df_posts['chunk'] == start_chunk]
for id in tqdm(df_chunk['id']):
    # append to csv file
    N_awards = get_n_awards(id)
    df_out = pd.DataFrame({'id':id,'N_awards':N_awards},index=[0])
    df_out.to_csv(p+file_name,mode='a',header=False,index=False)