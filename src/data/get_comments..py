import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import praw
from time import sleep

# Read-only instance
reddit_read_only = praw.Reddit(client_id="OlWj7Mu4aXh0eg",
                               client_secret="fIzRhpEeBYAwi8_i2hcyzoWwDnWOag",
                               user_agent="Scrapper")

url = "https://www.reddit.com/r/funny/comments/3g1jfi/buttons/"
submission = reddit.submission(url=url)