# imports
print('Starting..')
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import json
import torch 
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForPreTraining, AdamW, get_scheduler, get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ElectraModel
from transformers.utils import logging
logging.set_verbosity_error() #Remove warning msg - missing fine-tunning

import model_def as model_def
from dataset_def import Dataset

eval_dataloader = torch.load('eval_dataloader.pt')
train_dataloader = torch.load('train_dataloader.pt')
test_dataloader = torch.load('test_dataloader.pt')

# print length of data
print('Length of train data: ' + str(len(train_dataloader)))
print('Length of eval data: ' + str(len(eval_dataloader)))
print('Length of test data: ' + str(len(test_dataloader)))

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ' + str(device))

model = model_def.ElectraClassifier()
model = model.to(device)

# Set up optimizer and scheduler
EPOCHS = 50
optimizer = AdamW(model.parameters(), lr=1e-2, correct_bias=False,no_deprecation_warning=True)
total_steps = len(train_dataloader) * EPOCHS
loss_fn = nn.CrossEntropyLoss().to(device)

# evaluate model
def eval_model(model, data_loader, loss_fn, device, n_examples):
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

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)   

# train model
def train_epoch(model,data_loader,loss_fn,optimizer,device,n_examples):
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

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# predict
def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    ground_truth = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            network_features = d["network_features"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            network_features=network_features)

            _, preds = torch.max(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            ground_truth.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    ground_truth = torch.stack(ground_truth).cpu()

    return review_texts, predictions, prediction_probs, ground_truth

def save_history(history):

    train_acc = []
    for i in history['train_acc']:
        train_acc.append(i.cpu().detach().numpy())

    val_acc = []
    for i in history['val_acc']:
        val_acc.append(i.cpu().detach().numpy())

    vall_loss = []    
    train_loss = []
    for train_l, val_l in zip(history['train_loss'], history['val_loss']):
        train_loss.append(train_l)
        vall_loss.append(val_l)

    plt.figure(figsize=(10,8))
    plt.plot(train_acc, label='train accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('history_acurracy.png')
    plt.show()

    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='train loss')
    plt.plot(vall_loss, label='validation loss')
    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('history_loss.png')
    plt.show()

##### MAIN #####
## TRAIN
history = defaultdict(list)
best_accuracy = 0
for epoch in tqdm(range(EPOCHS)):

    train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device,len(train_dataloader.dataset))
    val_acc, val_loss = eval_model(model, eval_dataloader, loss_fn, device, len(eval_dataloader.dataset))

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# save history
save_history(history)

## predict ##
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_dataloader)
# save classification report
cr=pd.DataFrame(classification_report(y_test, y_pred, digits=4, output_dict=True))
cr.to_csv('classification_report.csv')

plt.figure(figsize=(10,8))
confusion_matrix = confusion_matrix(y_test, y_pred)
hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')

# print f1 score
try:
    print('F1 score: ', f1_score(y_test, y_pred, average='macro'))
except:
    print('no correct predictions!')
