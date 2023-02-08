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


# init
b_size = 8
EPOCHS = 78
learning_rate = 0.00964

# load data
df_train = pd.read_csv('~/data/train.csv')
df_test = pd.read_csv('~/data/test.csv')
df_eval = pd.read_csv('~/data/eval.csv')

print('Length of train data: ' + str(len(df_train)))
print('Length of eval data: ' + str(len(df_eval)))
print('Length of test data: ' + str(len(df_test)))
print('Total data: ' + str(len(df_train) + len(df_eval) + len(df_test)))

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
def eval_model(model, data_loader, device, n_examples):
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
def train_epoch(model,data_loader,optimizer,device,n_examples):
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
    plt.title('Training history', fontsize=23)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('history_acurracy_electra.png')
    plt.show()

    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='train loss')
    plt.plot(vall_loss, label='validation loss')
    plt.title('Training history', fontsize=23)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('history_loss_electra.png')
    plt.show()

def save_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.xticks([0.5,1.5], ['No Award', 'Award'], fontsize=20)
    plt.yticks([0.5,1.5], ['No Award', 'Award'],fontsize=20)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Actual', fontsize=20)
    plt.title('Confusion Matrix', fontsize=23)
    plt.savefig('confusion_matrix_electra.png')



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ' + str(device))
    model = model_def.ElectraClassifier()
    model = model.to(device)

    for param in model.electra.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss().to(device)

    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    train_dataloader = create_dataloader(df_train, tokenizer, 200, b_size)
    test_dataloader = create_dataloader(df_test, tokenizer, 200, b_size)
    eval_dataloader = create_dataloader(df_eval, tokenizer, 200, b_size)

    print('Batch size: ' + str(next(iter(eval_dataloader))['targets'].shape[0]))
    print('Learning rate: ' + str(learning_rate))
    print('Epochs: ' + str(EPOCHS))

    ## TRAIN
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in tqdm(range(EPOCHS)):

        train_acc, train_loss = train_epoch(model, train_dataloader, optimizer, device, len(train_dataloader.dataset))
        val_acc, val_loss = eval_model(model, eval_dataloader, device, len(eval_dataloader.dataset))

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state_electra.bin')
            best_accuracy = val_acc

    # save history
    save_history(history)

    ## predict ##
    y_text, y_pred, y_pred_probs, y_test = get_predictions(model,test_dataloader)

    # save classification report
    pd.DataFrame(classification_report(y_test, y_pred, digits=4, output_dict=True)).to_csv('classification_report_electra.csv')

    # save confusion matrix
    save_confusion_matrix(y_test, y_pred)

    # print f1 score
    print('F1 score: ', f1_score(y_test, y_pred, average='macro'))
