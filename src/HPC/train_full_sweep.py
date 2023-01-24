# imports
print('Starting..')
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import torch 
from torch import nn, optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import model_def as model_def
import model_def_full as model_def_full
from dataset_def import Dataset

# load data
df_train = pd.read_csv('~/data/train_full.csv')
df_test = pd.read_csv('~/data/test_full.csv')
df_eval = pd.read_csv('~/data/eval_full.csv')

# create dataloader
def create_dataloader(df, tokenizer, max_len, batch_size):
    ds = Dataset(
    network_features=df[['degree_cen', 'close_cen', 'activity', 'degree', 'N_nodes', 'N_edges','mentions','sentiment_compound','text_length']].to_numpy(),
    texts=df["text_title"].to_numpy(),
    targets=df['awarded'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    return DataLoader(ds,batch_size=batch_size,num_workers=2)

# evaluate model
def eval_model(model, data_loader, device, n_examples,loss_fn):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            targets = d["targets"].to(device)
            network_features = d["network_features"].to(device)

            outputs = model(network_features=network_features)
            loss = loss_fn(outputs, targets)      
            losses.append(loss.item())

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)

    return correct_predictions.double() / n_examples, np.mean(losses)   

# train model
def train_epoch(model,data_loader,optimizer,device,n_examples,loss_fn):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:

        targets = d["targets"].to(device)
        network_features = d["network_features"].to(device)

        outputs = model(network_features=network_features)
        
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / n_examples, np.mean(losses)


def main_train():
    run = wandb.init()

    learning_rate = run.config.lr
    b_size = run.config.batch_size
    EPOCHS = run.config.epochs

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss().to(device)

    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    train_dataloader = create_dataloader(df_train, tokenizer, 200, b_size)
    eval_dataloader = create_dataloader(df_eval, tokenizer, 200, b_size)

    best_accuracy = 0
    for epoch in tqdm(range(EPOCHS)):

        train_acc, train_loss = train_epoch(model, train_dataloader, optimizer, device, len(train_dataloader.dataset),loss_fn)
        val_acc, val_loss = eval_model(model, eval_dataloader, device, len(eval_dataloader.dataset),loss_fn)

        wandb.log(
                {
                    "Training_loss": train_loss,
                    "Validation_loss": val_loss,
                    "Training_accuracy": train_acc,
                    "Validation_accuracy": val_acc,
                })

        if val_acc > best_accuracy:
            best_accuracy = val_acc

if __name__ == "__main__":

    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64, 128]},
        'epochs': {'max': 50, 'min': 5},
        'lr': {'max': 0.1, 'min': 0.0001}
     }}

    # wandb
    key = 'e8d70bdabfe211a4d6306b5d0a8db41f77ebf3bd'
    wandb.login(key=key)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='reputation')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ' + str(device))
    model = model_def_full.ElectraClassifier()
    model = model.to(device)


    wandb.agent(sweep_id, function=main_train, count=4)