import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from os import listdir
from os.path import isfile, join
import re
import json
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    NNConv
)
from torch_geometric.typing import WITH_TORCH_CLUSTER

import copy

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

EPOCHS = 3000
IMAGE_SHOW_TRESHOLD = 0.05


def load_ARC(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/'):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
           with open(mypath+'/'+element,'r') as f:
            json_dict = json.load(f)
            for examples in json_dict['train']:


                    if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                        continue
                    m_i = examples['input']
                    connection_s = []
                    connection_t = []
                    for j in range(np.array(m_i).shape[1]):
                        for i in range(j,np.array(m_i).shape[0]*np.array(m_i).shape[1],np.array(m_i).shape[0]):
                            if i*np.array(m_i).shape[0] < np.array(m_i).shape[0]*np.array(m_i).shape[1]:
                                connection_s.append(i)
                                connection_t.append(i*np.array(m_i).shape[0])
                                
                    for j in range(np.array(m_i).shape[1]):
                        for i in range(j,np.array(m_i).shape[0]-1):
                                connection_s.append(j*np.array(m_i).shape[0]+i)
                                connection_t.append(j*np.array(m_i).shape[0]+i+1)

                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    
                    m_i = torch.tensor(m_i)
                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1],1))

                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = torch.tensor(m_o)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],1))

                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr

train_dataset = load_ARC(1280,mypath = './datasets/abstraction-and-reasoning-challenge/training/')
test_dataset = load_ARC(1280,mypath = './datasets/abstraction-and-reasoning-challenge/test/')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


num_features = 1
num_target = 1

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, num_features * 32),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 32, nn2, aggr='mean')

        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 32 * 64),
        )
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))


device = "cuda:0"
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for data in train_loader:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.mse_loss(model(data), data.y).backward()
        optimizer.step()

def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        
        pred = model(data).flatten()
    
        total_metric = (pred.shape[0]-torch.isclose(pred,data.y[0],rtol=0.05).flatten().sum())/pred.shape[0] 
        correct += total_metric 
    
        if total_metric <= IMAGE_SHOW_TRESHOLD:
            igure, axis = plt.subplots(1, 3)
            axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
            axis[ 0].set_title("Predicted")

            axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
            axis[ 1].set_title("True")
            
            axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
            axis[ 2].set_title("Input")
            plt.show()


    return correct / len(test_dataset)


for epoch in range(1, EPOCHS):
    train(epoch)
    test_acc = test()
    print(f'Epoch: {epoch:02d}, Test: {test_acc:.4f}')
