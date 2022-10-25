from time import sleep
import pandas as pd
from tqdm import tqdm
from psaw import PushshiftAPI
import datetime as dt
import warnings
warnings.filterwarnings("ignore")


def get_periods(tstart, tend, interval):
    periods = []

    period_start = tstart
    while period_start < tend:
        period_end = min(period_start + interval, tend)
        periods.append((
            int(period_start.timestamp()), 
            int(period_end.timestamp())
            ))
        period_start = period_end

    return periods


def convert_utc_to_date(df):
    df['date'] = pd.to_datetime(df['created_utc'],unit='s')
    return df

def download_comments(start,end,subreddit,folder_name,file_name,limit,check_point):

    def data_prep_comments(subreddit, start_time, end_time, filters, limit):
        if (len(filters) == 0):
            filters = ['id', 'author', 'created_utc',
                    'body', 'subreddit','score','parent_id','post_id']
                    #We set by default some usefull columns 

        comments = list(api.search_comments(
            subreddit=subreddit,    #Subreddit we want to audit
            after=start_time,       #Start date
            before=end_time,        #End date
            filter=filters,         #Column names we want to retrieve
            limit=limit))           #Max number of comments
        return pd.DataFrame([thing.d_ for thing in comments]) #Return dataframe for analysis


    if file_name == None: file_name = subreddit

    if check_point:
        print('Continuing from last checkpoint..')
        current_df = pd.read_csv("./data/raw/" + folder_name + '/' + file_name + ".csv")
        current_df.columns = ['author', 'body', 'created_utc', 'id', 'parent_id','score', 'subreddit', 'created']
        current_df = convert_utc_to_date(current_df)
        start = current_df.created.max()
    else: 
        print('Starting from scratch..')
        pd.DataFrame().to_csv("./data/raw/" + folder_name + '/' + file_name + ".csv", index=False, header=False)
    
    print('Downloading to..: ', file_name+'.csv')
    print('Start date: ' + str(start))
    print('End date: ' + str(end))
    print('Subreddit: ' + subreddit)
    
    print('Setting up API..')
    api = PushshiftAPI()

    periods = get_periods(start, end, dt.timedelta(days=2))

    for period in tqdm(periods):
        df_c = data_prep_comments(subreddit, start_time=period[0], end_time=period[1], filters=[], limit=limit)
        df_c.to_csv("./data/raw/" + folder_name + '/' +  file_name + ".csv", mode='a', index=False, header=False)
        # wait N second to avoid rate limit
        sleep(2)