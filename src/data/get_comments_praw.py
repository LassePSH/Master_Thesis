import pandas as pd
import praw
from time import sleep
from tqdm import tqdm
tqdm.pandas(desc="Getting comments")


# Read-only instance
def get_reddit_instance():
    reddit = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                                client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                                user_agent="Scrapper")
    return reddit

def empty_file(file_name,folder_name):
    pd.DataFrame(columns=['post_id','parent_id','author','id','body','created','score']).to_csv("./data/raw/" + folder_name + '/' + file_name + ".csv", index=False, header=True)

# read posts data
def read_posts(subreddit_file_name ,folder_name):
    posts_df = pd.read_csv("./data/raw/" + folder_name + '/' + subreddit_file_name + ".csv")
    posts_df.columns = ['author','created_utc','domain','id','n_comments','score','text','title','url','date']
    return posts_df

# get comments
def get_comments_from_id(id,file_name,folder_name,reddit):
    cache_dict = {'post_id':[],'parent_id':[],'author':[],'id':[],'body':[],'created':[],'score':[]}
    
    submission = reddit.submission(id=id)
    submission.comments.replace_more(limit=None)
    comment_queue = submission.comments[:] 
    while comment_queue:
        comment = comment_queue.pop(0)

        cache_dict['post_id'].append(id)
        cache_dict['parent_id'].append(comment.parent_id)
        cache_dict['author'].append(comment.author)
        cache_dict['id'].append(comment.id)
        cache_dict['body'].append(comment.body)
        cache_dict['created'].append(comment.created_utc)
        cache_dict['score'].append(comment.score)
    

        comment_queue.extend(comment.replies)
        sleep(0.5)

    pd.DataFrame(cache_dict).to_csv("./data/raw/" + folder_name + '/' + file_name + ".csv", mode='a', index=False, header=False)


### MAIN ###
def download_comments(posts_file_name,comments_file_name,folder_name):
    print("Starting reddit instance...")
    reddit = get_reddit_instance()
    print("Reading posts...")
    posts_df = read_posts(posts_file_name,folder_name)
    print("Creating empty file...")
    empty_file(comments_file_name,folder_name)
    print("Getting comments...")
    posts_df['id'].progress_apply(lambda x: get_comments_from_id(x,comments_file_name,folder_name,reddit))