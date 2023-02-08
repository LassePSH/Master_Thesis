import torch 
from torch import nn
import torch.nn.functional as F
from transformers import ElectraModel

class ElectraClassifier(nn.Module):
    def __init__(self,num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels

        # text features
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.dense_txt = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size) # 256
        self.dropout_txt = nn.Dropout(self.electra.config.hidden_dropout_prob)

        # combined features
        self.dense_cat1 = nn.Linear(in_features=(256+12),out_features=512) # 256 from text features + 12 from network features 
        self.dense_cat2 = nn.Linear(in_features=512,out_features=1024)
        self.dense_cat3 = nn.Linear(in_features=1024,out_features=2048)
        self.dense_cat6 = nn.Linear(in_features=2048,out_features=1024)
        self.dense_cat7 = nn.Linear(in_features=1024,out_features=512)
        self.dense_cat8 = nn.Linear(in_features=512,out_features=256)

        # output layer
        self.out_proj = nn.Linear(256, self.num_labels) # 2 labels

    def classifier(self,sequence_output,network_features):
        # text features
        x_txt = sequence_output[:, 0, :] #[CLS] token
        x_txt = F.relu(self.dense_txt(x_txt))
        x_txt = self.dropout_txt(x_txt)
        
        # combined features
        x = torch.cat((x_txt,network_features),dim=1) 
        x = F.relu(self.dense_cat1(x))
        x = F.relu(self.dense_cat2(x))
        x = F.relu(self.dense_cat3(x))
        x = F.relu(self.dense_cat6(x))
        x = F.relu(self.dense_cat7(x))
        x = F.relu(self.dense_cat8(x))

        # output layer
        logits = self.out_proj(x)
        return logits

    def forward(self, input_ids=None,attention_mask=None,network_features=None):
        discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output,network_features)

        return logits