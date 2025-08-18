
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential,Conv1d,MultiheadAttention,ConvTranspose1d


import numpy as np

from torch_geometric.nn import (
    NNConv,
    GraphConv   
)

class CNN(torch.nn.Module):
    def __init__(self,num_features = 10,num_target = 10,local_device = 'cuda:0'):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')


        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))

        data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
 
        data.x = F.elu(self.cnn(torch.reshape(data.x, (1,-1))))
        data.x = F.elu(self.cnnT(torch.reshape(data.x, (1,-1))))
        data.x = torch.reshape(data.x, (-1,1))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))

class Attention(torch.nn.Module):
    def __init__(self,num_features = 10,num_target = 10,local_device = 'cuda:0'):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        
        nn_start = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25,num_features),
        )
        self.nn_middle = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, num_features),
        )
        nn_end = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25,1* 64),
        )

        self.conv1 = NNConv(num_features, num_features,nn=nn_start, aggr='mean')
        self.conv_m = NNConv(num_features, num_features,nn=self.nn_middle, aggr='mean')
        self.conv2 = NNConv(num_features, 64,nn=nn_end, aggr='mean')
        self.multihead_attn = MultiheadAttention(num_features, 1)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
       data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
       data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
       data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))


       x = F.elu(self.fc1(data.x))
       x = F.dropout(x, training=self.training)
       return F.elu(self.fc2(x))


class HotEncoding(torch.nn.Module):
    
    def __init__(self,num_features = 10,num_target = 10,local_device = 'cuda:0'):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        

        self.grap_conv = GraphConv(self.num_features, 10, aggr='mean')
        self.grap_conv_end = GraphConv(self.num_features, 10, aggr='mean')


        self.fc1 = torch.nn.Linear(10, 128)
        self.fc2 = torch.nn.Linear(128, self.num_target)
        
        self.v1 = torch.tensor( np.array([[1,0,0]]),dtype=torch.float).to(self.local_device)
    def forward(self, data):
 

        data.x = F.elu(self.grap_conv(data.x, data.edge_index))
        data.x = F.elu(self.grap_conv_end(data.x, data.edge_index))

        x = F.elu(self.fc1(data.x))
        return F.softmax(self.fc2(x))