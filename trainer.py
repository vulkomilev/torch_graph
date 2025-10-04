
import torch.nn.functional as F
import copy

def train_mse_globalNode(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()

    if data is None:
        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data)[:-10*param_dict['hidden_node_count']], data.y)
            loss.backward()
            optimizer.step()
        mlflow.log_metric(key="loss",value=loss,step=epoch)
    else:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.mse_loss(model(data), data.y).backward()
        optimizer.step()

def train_mse_Neighbor_Sampling(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()

    if data is None:
        print("loader",loader)
        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            optimizer.step()
        mlflow.log_metric(key="loss",value=loss,step=epoch)
    else:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.mse_loss(model(data), data.y).backward()
        optimizer.step()

def train_mse(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()

    if data is None:
        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            optimizer.step()
        mlflow.log_metric(key="loss",value=loss,step=epoch)
    else:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.mse_loss(model(data), data.y).backward()
        optimizer.step()

def train_cross_entropy(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()

    if data is None:

        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(data), data.y).backward()
            optimizer.step()
        for data in loader:
            data = copy.deepcopy(data)
            data = data.to(device)
        mlflow.log_metric(key="loss",value=F.cross_entropy(model(data), data.y),step=epoch)
            
    else:
        data = copy.deepcopy(data)
        data = data.to(device)
        optimizer.zero_grad()
        F.cross_entropy(model(data), data.y).backward()
        optimizer.step()