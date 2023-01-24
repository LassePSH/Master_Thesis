import torch 
from torch import nn
import torch.nn.functional as F

class ElectraClassifier(nn.Module):
    def __init__(self,num_labels=2):
        super(ElectraClassifier,self).__init__()
        self.num_labels = num_labels

        # network features
        self.network_input = nn.Linear(in_features=9,out_features=2048) # 9 network features
        self.dense_net2 = nn.Linear(in_features=2048,out_features=2048)
        self.dense_net3 = nn.Linear(in_features=2048,out_features=2048)
        self.dense_net4 = nn.Linear(in_features=2048,out_features=2048)

        # output layer
        self.out_proj = nn.Linear(2048, self.num_labels)

    def forward(self,network_features=None):
        x_net = F.gelu(self.network_input(network_features))
        x_net = F.gelu(self.dense_net2(x_net))
        x_net = F.gelu(self.dense_net3(x_net))
        x_net = F.gelu(self.dense_net4(x_net))

        logits = self.out_proj(x_net)
        
        return F.softmax(logits,dim=1)