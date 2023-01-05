import pandas as pd
import praw
from time import sleep
from tqdm import tqdm

tqdm.pandas(desc="Getting Karma")


# Read-only instance
def get_reddit_instance():
    print("starting reddit instance...")
    reddit = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                                client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                                user_agent="Scrapper")
    return reddit

#download files
def get_authors(file_name):
    print('Getting authors from file: ' + file_name)
    df=pd.read_csv("./data/raw/" + file_name + ".csv")
    df.columns = ['author','created_utc','domain','id','n_comments','text','title','url','date']
    
    print('Getting authors from file: ' + file_name+"_comments")
    df_comments=pd.read_csv("./data/raw/" + file_name + "_comments.csv")

    print("Concatenating...")
    df_all_nodes = pd.DataFrame()
    df_all_nodes['author'] = pd.concat([df_comments["author"].drop_duplicates(), df["author"].drop_duplicates()])
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes['author'].dropna(inplace=True)
    df_all_nodes.drop(df_all_nodes.loc[df_all_nodes['author']=='[deleted]'].index, inplace=True)
    df_all_nodes = df_all_nodes[df_all_nodes["author"].apply(lambda x: isinstance(x, str))]
    
    return df_all_nodes


def empty_file(file_name):
    print("Creating empty file...")
    pd.DataFrame(columns=['author','karma']).to_csv("./data/raw/" + file_name + "_karma.csv", index=False, header=True)


def get_karma(author,reddit):
    cache_dict = {'author':[],'karma':[]}
    
    try:
        user = reddit.redditor(author)
        karma = user.link_karma
    except:
        karma = None
        print("Error getting karma for user: " + author)
    
    cache_dict['author'].append(author)
    cache_dict['karma'].append(karma)

    pd.DataFrame(cache_dict).to_csv("./data/raw/" + file_name + "_karma.csv", mode='a', index=False, header=False)


### MAIN ###
file_name = 'jazznoir_2015_2022'

reddit = get_reddit_instance()
df_all_nodes = get_authors(file_name)

empty_file(file_name)

print('getting karma...')
df_all_nodes['author'].progress_apply(lambda x: get_karma(x,reddit))

