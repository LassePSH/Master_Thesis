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

        # network features
        self.dense_net = nn.Linear(in_features=7,out_features=256) # 7 network features
        # self.dense_net2 = nn.Linear(in_features=20,out_features=1024)
        # self.dense_net3 = nn.Linear(in_features=1024,out_features=2048)
        # self.dense_net4 = nn.Linear(in_features=2048,out_features=2048)
        # self.dense_net5 = nn.Linear(in_features=2048,out_features=2048)
        # self.dense_net6 = nn.Linear(in_features=2048,out_features=120)

        # combined features
        self.dense_cat1 = nn.Linear(in_features=512,out_features=1024)
        # self.dense_cat2 = nn.Linear(in_features=1024,out_features=1024)
        self.dense_cat3 = nn.Linear(in_features=1024,out_features=512)
        self.dropout_cat = nn.Dropout()

        # output layer
        self.out_proj = nn.Linear(512, self.num_labels)

    def classifier(self,sequence_output,network_features):
        x_txt = sequence_output[:, 0, :]
        x_txt = self.dropout_txt(x_txt)
        x_txt = F.gelu(self.dense_txt(x_txt))

        x_net = F.gelu(self.dense_net(network_features))
        # x_net = F.gelu(self.dense_net2(x_net))
        # x_net = F.gelu(self.dense_net3(x_net))
        # x_net = F.gelu(self.dense_net4(x_net))
        # x_net = F.gelu(self.dense_net5(x_net))
        # x_net = F.gelu(self.dense_net6(x_net))
        
        x = torch.cat((x_txt,x_net),dim=1) #256 from text features + 256 from network features = 512
        x = F.gelu(self.dense_cat1(x))
        # x = F.gelu(self.dense_cat2(x))
        x = F.gelu(self.dense_cat3(x))
        x = self.dropout_cat(x)

        logits = self.out_proj(x)
        sm = nn.Softmax(dim=1)
        return sm(logits)

    def forward(self, input_ids=None,attention_mask=None,network_features=None):
        discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]

        logits = self.classifier(sequence_output,network_features)
        
        return logits
