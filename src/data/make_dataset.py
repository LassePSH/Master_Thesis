import get_comments
import get_posts
import datetime as dt
import os

subreddit = input("Input subreddit: ")
print('')

start=dt.datetime(year=2015, month=1, day=1)
end=dt.datetime(year=2022, month=1, day=1)


folder_name = subreddit+'_'+str(start)[:10]+'-'+str(end)[:10]
post_file_name=subreddit+'_posts'
comments_file_name=subreddit+'_comments'

newpath = r'./data/raw/' + folder_name
if not os.path.exists(newpath):
    print('Setting up new path..')
    os.makedirs(newpath)
else: print('Path already exists..')

print('')
print('folder_name: ', folder_name)
print('')

# check if files exists
# if os.path.isfile("./data/raw/" + folder_name + '/' + post_file_name + ".csv"):
#     print('Post file already exists..')
#     print('changes file name to: ' + post_file_name + '_new.csv\n')
#     post_file_name = post_file_name + '_new'

# if os.path.isfile("./data/raw/" + folder_name + '/' + comments_file_name + ".csv"):
#     print('Comments file already exists..')
#     print('changes file name to: ' + comments_file_name + '_new.csv \n')
#     comments_file_name = comments_file_name + '_new'


# ## fire up the engines
# print('Runnin get_posts..')
# get_posts.download_posts(
#             start=start,
#             end=end,
#             subreddit=subreddit,
#             folder_name=folder_name,
#             file_name=post_file_name,
#             limit=None,
#             check_point=True)
# print('Posts downloaded..\n')

print('Running get_comments..')
get_comments.download_comments(
            start=start,
            end=end,
            subreddit=subreddit,
            folder_name=folder_name,
            file_name=comments_file_name,
            limit=None,
            check_point=True)

print('Comments downloaded..\n')
print('DONE!')