import os.path as osp
import matplotlib.pyplot as plt
import torch
from torch import nn
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
from torch_geometric.nn.conv import (
    GraphConv,NNConv,GraphConvSymbolic
)
from torch_geometric.typing import WITH_TORCH_CLUSTER

import copy
import mlflow
from mlflow.models import infer_signature

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

EPOCHS = 40000
IMAGE_SHOW_TRESHOLD = 0.15

num_features = 1
num_target = 1
dim_log = {}

difficult = []

loaded_names = []

def create_or_add_log(dim, score):
    if dim not in  list(dim_log.keys()):
        dim_log[dim] = []
    dim_log[dim].append(score)

def eval_log():
    keys = list(dim_log.keys())
    keys.sort()
    for dim in  keys:
        print(dim,sum(dim_log[dim])/len(dim_log[dim]),len(dim_log[dim]))

all_metric = []

#@step
def load_ARC(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/'):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
          #print('element',element)
          #print(difficult_names)
          #if element in difficult_names:
           with open(mypath+'/'+element,'r') as f:
           
            json_dict = json.load(f)
            for examples in json_dict['train']:


                    if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                        continue
                    m_i = examples['input']
                    connection_s = []
                    connection_t = []
                    features = []
                    for j in range(np.array(m_i).shape[1]):
                        for i in range(j,np.array(m_i).shape[0]*np.array(m_i).shape[1],np.array(m_i).shape[0]):
                            if i*np.array(m_i).shape[0] < np.array(m_i).shape[0]*np.array(m_i).shape[1]:
                                connection_s.append(i)
                                connection_t.append(i*np.array(m_i).shape[0])
                                features.append(1)
                                
                    for j in range(np.array(m_i).shape[1]):
                        for i in range(j,np.array(m_i).shape[0]-1):
                                connection_s.append(j*np.array(m_i).shape[0]+i)
                                connection_t.append(j*np.array(m_i).shape[0]+i+1)
                                features.append(1)

                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    
                    m_i = torch.tensor(m_i)
                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1],1))
                    m_i = m_i.expand((-1,num_features))
                    
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
                                            file_name = element,
                                            edge_attr= torch.tensor(np.ones(np.array([connection_s]).shape).transpose(),dtype=torch.float)))

    return return_arr






class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
    
        def symbolic_fun(x_j):
            
            #if x_j[0]>0.5:
            #    #print(x_j)
            #    x_j[0] = 1
            return x_j
    

        #self.conv1 = GraphConv(num_features, num_features, aggr='add')
        #self.conv_m = GraphConv(num_features, num_features, aggr='add')
        #self.conv2 = GraphConv(num_features, 64,  aggr='add')
        #GraphConvSymbolic(num_features, num_features,symbolic_function=symbolic_fun, aggr='add')
        self.conv1 = NNConv(num_features, num_features,nn=nn_start, aggr='mean')
        self.conv_m = NNConv(num_features, num_features,nn=self.nn_middle, aggr='mean')
        self.conv2 = NNConv(num_features, 64,nn=nn_end, aggr='mean')
        self.multihead_attn = nn.MultiheadAttention(num_features, 1)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
       data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
       data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
      # data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
      # data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
    #    data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
    #    data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
       data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        # data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

       x = F.elu(self.fc1(data.x))
       x = F.dropout(x, training=self.training)
       return F.elu(self.fc2(x))


device = "cuda:0"
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch,dataset):
    model.train()
    for data in dataset:
        
        #data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        #print(model(data).shape,data.y.shape)
        
        F.mse_loss(model(data), data.y).backward()
        optimizer.step()
loss_log  = []
def test(epoch,dataset,show_images=False):
    model.eval()
    correct = 0

    for data in test_loader:
        #print("data1",data)
        data_second = copy.deepcopy(data)
        data = data.to(device)
        
        pred = model(data).flatten()

        total_metric = (pred.shape[0]-torch.isclose(pred,data.y.flatten(),rtol=0.05).flatten().sum())/pred.shape[0]
        all_metric.append(total_metric)

        #print('total_metric',total_metric.cpu().detach().numpy())
        mlflow.log_metric("min", min(all_metric))
        mlflow.log_metric("test_loss", min(all_metric))
        create_or_add_log(pred.shape[0],total_metric)
        correct += total_metric 
    
        if show_images and (total_metric <= IMAGE_SHOW_TRESHOLD and epoch > 200):
            igure, axis = plt.subplots(1, 3)
            axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
            axis[ 0].set_title("Predicted")

            axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
            axis[ 1].set_title("True")
            
            axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
            axis[ 2].set_title("Input")
            plt.show()

        if total_metric >= IMAGE_SHOW_TRESHOLD or epoch > 200:
            difficult.append(data_second.file_name)

    return correct / len(test_dataset)
train_dataset = load_ARC(1280,mypath = './datasets/abstraction-and-reasoning-challenge/training/')
test_dataset = load_ARC(1280,mypath = './datasets/abstraction-and-reasoning-challenge/test/')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def pipeline():

    #mlflow.set_experiment(experiment_id="0")

    #mlflow.autolog()
    for epoch in range(1, EPOCHS):
        train(epoch,train_loader)
        test_acc = test(epoch,test_loader,True)
        print(f'Epoch: {epoch:02d}, Test: {test_acc:.4f} , Min:{min(all_metric):.4f}')
#print(difficult)
#difficult_loader = DataLoader(difficult, batch_size=1, shuffle=False)
# model = Net().to(device)

# for epoch in range(1, EPOCHS):
#     train(epoch,difficult_loader)
#     test_acc = test(epoch,difficult_loader,True)
#     print(f'Epoch: {epoch:02d}, Test: {test_acc:.4f} , Min:{min(all_metric):.4f}')
#mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    #mlflow.autolog()
    pipeline()
eval_log()