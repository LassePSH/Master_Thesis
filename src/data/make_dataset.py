import get_comments
import get_posts
import datetime as dt
import os

start=dt.datetime(year=2015, month=1, day=1)
end=dt.datetime(year=2015, month=1, day=10)
subreddit='wallstreetbets'


folder_name = subreddit+'_'+str(start)[:10]+'-'+str(end)[:10]
file_name=subreddit+'_posts'
comments_file_name=subreddit+'_comments'

newpath = r'./data/raw/' + folder_name
if not os.path.exists(newpath):
    print('Setting up new path..')
    os.makedirs(newpath)
else: print('Path already exists..')


## fire up the engines
get_posts.download_posts(
            start=start,
            end=end,
            subreddit=subreddit,
            folder_name=folder_name,
            file_name=file_name, # File name!
            limit=None,
            check_point=False)

print('Posts downloaded..')

get_comments.download_comments(
                            posts_file_name=file_name,
                            comments_file_name=comments_file_name,
                            folder_name=folder_name)

# To Do Get score from posts!!!!!!