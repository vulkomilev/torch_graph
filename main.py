
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER
import copy
from loader import load_ARC_onehot
from models import HotEncoding
from trainer import train_cross_entropy
from tester import test

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

EPOCHS = 20000
TRAIN_EPOCHS = 10
IMAGE_SHOW_TRESHOLD = 0.01
total_metric_min = 1.0
local_device = 'cuda:0'

train_dataset = load_ARC_onehot(30,mypath = './datasets/abstraction-and-reasoning-challenge/training/')
test_dataset = load_ARC_onehot(30,mypath = './datasets/abstraction-and-reasoning-challenge/test/')

print('train_dataset',train_dataset[0].x.shape)
print('test_dataset',test_dataset[0].x.shape)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

model = HotEncoding().to(local_device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, EPOCHS):
     train_cross_entropy(epoch,train_dataset,optimizer,model,local_device,None)
# torch.save(model.state_dict(), "./model")
checkpoint = torch.load("./model", weights_only=False)
model.load_state_dict(checkpoint)
for epoch in range(1, EPOCHS):
     test_acc = test(epoch,model,test_loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_cross_entropy,local_device)
     print(f'Epoch: {epoch:02d}, Test: {test_acc:.4f}, Min: {total_metric_min:.4f}')
