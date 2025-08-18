
import torch.nn.functional as F
import copy

def train_mse(epoch,loader,optimizer,model,device,data = None):
    model.train()

    if data is None:
        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
            optimizer.zero_grad()
            F.mse_loss(model(data), data.y).backward()
            optimizer.step()
    else:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.mse_loss(model(data), data.y).backward()
        optimizer.step()

def train_cross_entropy(epoch,loader,optimizer,model,device,data = None):
    model.train()

    if data is None:
        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(data), data.y).backward()
            optimizer.step()
    else:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.cross_entropy(model(data), data.y).backward()
        optimizer.step()