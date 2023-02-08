import torch 
from torch import nn
import torch.nn.functional as F

class ElectraClassifier(nn.Module):
    def __init__(self,num_labels=2):
        super(ElectraClassifier,self).__init__()
        self.num_labels = num_labels

        # network + text features
        self.network_input = nn.Linear(in_features=14,out_features=256)
        self.dense_net2 = nn.Linear(in_features=256,out_features=512)
        self.dense_net3 = nn.Linear(in_features=512,out_features=1024)
        self.dense_net4 = nn.Linear(in_features=1024,out_features=2048)
        self.dense_net5 = nn.Linear(in_features=2048,out_features=1024)
        self.dense_net6 = nn.Linear(in_features=1024,out_features=512)
        self.dense_net7 = nn.Linear(in_features=512,out_features=256)

        # output layer
        self.out_proj = nn.Linear(256, self.num_labels)

    def forward(self,network_features=None):
        x_net = F.relu(self.network_input(network_features))
        x_net = F.relu(self.dense_net2(x_net))
        x_net = F.relu(self.dense_net3(x_net))
        x_net = F.relu(self.dense_net4(x_net))
        x_net = F.relu(self.dense_net5(x_net))
        x_net = F.relu(self.dense_net6(x_net))
        x_net = F.relu(self.dense_net7(x_net))

        logits = self.out_proj(x_net)
        return logits