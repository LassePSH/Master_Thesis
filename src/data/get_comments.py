import pandas as pd
import praw
# from time import sleep
from tqdm import tqdm
tqdm.pandas(desc="Getting comments")


# Read-only instance
def get_reddit_instance():
    reddit = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                                client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                                user_agent="Scrapper")
    return reddit

def empty_file(file_name):
    pd.DataFrame(columns=['post_id','parent_id','author','id','body','created']).to_csv("./data/raw/" + file_name + ".csv", index=False, header=True)

# read posts data
def read_posts(subreddit_file_name):
    posts_df = pd.read_csv("./data/raw/"+ subreddit_file_name + ".csv")
    posts_df.columns = ['author','created_utc','domain','id','n_comments','text','title','url','date']
    return posts_df

# get comments
def get_comments_from_id(id,file_name):
    cache_dict = {'post_id':[],'parent_id':[],'author':[],'id':[],'body':[],'created':[]}
    
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
    

        comment_queue.extend(comment.replies)
    
    pd.DataFrame(cache_dict).to_csv("./data/raw/" + file_name + ".csv", mode='a', index=False, header=False)


### MAIN ###
comments_file_name = 'jazznoir_2015_2022_comments' # new file name
posts_file_name = 'jazznoir_2015_2022' # old file name

# reddit instance
reddit=get_reddit_instance()

# create enmpty file
empty_file(comments_file_name)

# read posts
posts_df=read_posts(posts_file_name)

# get comments
print('Getting comments...')
print('from: ', posts_file_name+'.csv')
print('to: ', comments_file_name+'.csv')
posts_df.id.progress_apply(lambda x: get_comments_from_id(x,comments_file_name))