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
from dataset_def import Dataset

# load data
df_train = pd.read_csv('~/data/train.csv')
df_test = pd.read_csv('~/data/test.csv')
df_eval = pd.read_csv('~/data/eval.csv')

# create dataloader
def create_dataloader(df, tokenizer, max_len, batch_size):
    ds = Dataset(
    network_features=df[['degree_cen', 'close_cen', 'activity', 'degree', 'N_nodes', 'N_edges','mentions',
                        'frac_rec','N_rec','degree_in','degree_out','N_rec_author']].to_numpy(),
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
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            network_features = d["network_features"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            network_features=network_features)

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
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        network_features = d["network_features"].to(device)

        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        network_features=network_features)
        
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_def.ElectraClassifier()
    model = model.to(device)

    # for param in model.electra.parameters():
    #     param.requires_grad = False

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
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                })

        if val_acc > best_accuracy:
            best_accuracy = val_acc

if __name__ == "__main__":

    sweep_configuration = {
    'method': 'bayes',
    'name': 'Electra_no_freeze',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [8,16, 32, 64, 128]},
        'epochs': {'max': 150, 'min': 10},
        'lr': {'max': 0.1, 'min': 0.0001}
     }}

    wandb
    key = 'e8d70bdabfe211a4d6306b5d0a8db41f77ebf3bd'
    wandb.login(key=key)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='reputation')
    wandb.agent(sweep_id, function=main_train, count=20)