import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import datetime 

folder_name = input("Folder name: ")

files = os.listdir('/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+folder_name)
files = [file for file in files if file[-8:] == '.gpickle']
print("Number of .gpickle files : ", len(files))

##################### Graph Analysis #####################

def get_before_date(file_name):
    return datetime.datetime.strptime(file_name[17:27], '%Y-%m-%d')
    

files_date_dict = {}

for file in files:
    date = get_before_date(file)
    if date in files_date_dict:
        "Error: date already in dict"
    else:
        files_date_dict[date] = [file]

# sort dict by date

files_date_dict = {k: v for k, v in sorted(files_date_dict.items(), key=lambda item: item[0])}

G = nx.Graph()

# make new folder
new_folder_name = folder_name + '_joined'
os.mkdir('/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+new_folder_name)

for files in tqdm(files_date_dict.values()):
    G_n=nx.read_gpickle('/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+folder_name+'/' +files[0])
    G = nx.compose(G,G_n)
    # save graph
    nx.write_gpickle(G, '/home/pelle/Master_Thesis/data/processed/wallstreetbets_temporal_graphs/'+new_folder_name+'/' +'graph_'+files[0][17:27]+'.gpickle')