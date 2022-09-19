from threading import current_thread
from time import sleep
import pandas as pd
from tqdm import tqdm
from psaw import PushshiftAPI
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

def convert_utc_to_date(df):
    df['date'] = pd.to_datetime(df['created_utc'],unit='s')
    return df

def data_prep_posts(subreddit, start_time, end_time,  limit,api):
    filters = ['id', 'author', 'created_utc',
                'domain', 'url',
                'title', 'num_comments','selftext']                 
                #We set by default some useful columns

    posts = list(api.search_submissions(
        subreddit=subreddit,   #Subreddit we want to audit
        after=start_time,      #Start date
        before=end_time,       #End date
        filter=filters,        #Column names we want to retrieve
        limit=limit))          ##Max number of posts

    return pd.DataFrame([thing.d_ for thing in posts])


def create_dataset(start,end,subreddit,name,limit,check_point):
    if name == None: name = subreddit

    if check_point:
        print('Continuing from last checkpoint..')
        current_df = pd.read_csv("./data/raw/" + name + ".csv")
        current_df.columns = ['author','created_utc','domain','id','n_comments','text','title','url','date']
        current_df = convert_utc_to_date(current_df)
        start = current_df.date.max()
    else: 
        print('Starting from scratch..')
        pd.DataFrame().to_csv("./data/raw/" + name + ".csv", index=False, header=False)
    
    delta = end - start
    print('Downloading to..: ', name+'.csv')
    print('Start date: ' + str(start))
    print('End date: ' + str(end))
    print('Subreddit: ' + subreddit)
    
    print('Setting up API..')
    api = PushshiftAPI()

    print('Starting..') 
    for d in tqdm(range(delta.days + 1)):
        d_1=dt.timedelta(days=1)
        d_n=dt.timedelta(days=d+1)
        start_get=int((start+d_n-d_1).timestamp())
        end_get=int((start+d_n).timestamp())
        
        df=data_prep_posts(subreddit,start_get,end_get,limit,api)
        df.to_csv("./data/raw/" + name + ".csv", mode='a', index=False, header=False)
        # wait 1 second to avoid rate limit
        sleep(1)





start=dt.datetime(year=2013, month=1, day=1)
end=dt.datetime(year=2022, month=1, day=1)

create_dataset(
            start=start,
            end=end,
            subreddit='wallstreetbets',
            name='wallstreetbets_2018_2022', # File name!
            limit=None,
            check_point=True)