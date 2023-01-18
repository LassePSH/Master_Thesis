import os 
import pandas as pd
import numpy as np
import torch 
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
# from transformers import ElectraModel

# Load Data
p = '/home/pelle/Master_Thesis/data/raw/wallstreetbets/graph_features_2/'
file_names=os.listdir(p)

data = []
for name in file_names:
    data.append(pd.read_csv(p+name))

df_gf=pd.concat(data)
df_balanced = pd.read_csv('/home/pelle/Master_Thesis/data/raw/wallstreetbets/balanced_data_chunked10.csv')

df_gf.drop_duplicates(inplace=True)
df=df_balanced.join(df_gf.set_index('id'),on='id')
df.dropna(subset='degree',inplace=True)
df=df[['author', 'date', 'score', 'n_comments', 'id',
       'n_awards', 'text_title', 'degree_cen', 'close_cen', 
       'activity', 'degree', 'N_nodes', 'N_edges','mentions']]

df.loc[df.n_awards==0,'awarded']=0
df.loc[df.n_awards!=0,'awarded']=1

df['n_comments']=df.n_comments.apply(lambda x: np.sqrt(x**2))
df['date'] = pd.to_datetime(df['date'])
df['awarded'] = df['awarded'].astype(int)

# shuffle order of df
df = df.sample(frac = 1)

print('Data loaded')
print(df.shape)
print()


class Dataset():
  def __init__(self, texts, targets, tokenizer, max_len,network_features):
    self.network_features = network_features
    self.text = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.text)

  def __getitem__(self, item):
    network_features = self.network_features[item]
    text = str(self.text[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
        'network_features': torch.tensor(network_features, dtype=torch.float),
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)}

def create_dataloader(df, tokenizer, max_len, batch_size):
  ds = Dataset(
    network_features=df[['degree_cen', 'close_cen', 'activity', 'degree', 'N_nodes', 'N_edges','mentions']].to_numpy(),
    texts=df["text_title"].to_numpy(),
    targets=df['awarded'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)

  return DataLoader(ds,batch_size=batch_size,num_workers=2)

tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')

train_dataloader = create_dataloader(df, tokenizer, 200, 20)
print('Dataloader created')
print(type(train_dataloader))
print(iter(train_dataloader).next().keys())

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=42)

train_dataloader = create_dataloader(df_train, tokenizer, 200, 8)
test_dataloader = create_dataloader(df_test, tokenizer, 200, 8)
eval_dataloader = create_dataloader(df_eval, tokenizer, 200, 8)

# save
torch.save(train_dataloader, '/home/pelle/Master_Thesis/data/processed/dataloaders/week10/train_dataloader.pt')
torch.save(test_dataloader, '/home/pelle/Master_Thesis/data/processed/dataloaders/week10/test_dataloader.pt')
torch.save(eval_dataloader, '/home/pelle/Master_Thesis/data/processed/dataloaders/week10/eval_dataloader.pt')