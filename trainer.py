import torch.nn
import torch.nn.functional as F
import copy
import numpy as np

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

def train_mse_autoencoder(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()

    if data is None:
        for data in loader:
            data = copy.deepcopy(data)
            data["input"] = data["x"].to(device)
            data["output"] = data["y"].to(device)
            
            optimizer.zero_grad()
            loss = F.mse_loss(model(data["input"]), data["output"])
            loss.backward()
            optimizer.step()
        mlflow.log_metric(key="loss",value=loss,step=epoch)
    else:
        data = copy.deepcopy(data)
        data["input"] = data.to(device["input"])
        data["output"] = data.to(device["output"])
        optimizer.zero_grad()
        F.mse_loss(model(data["input"]), data["output"]).backward()
        optimizer.step()

def train_crossentr_autoencoder_cat(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()

    if data is None:
        for data in loader:
            data = copy.deepcopy(data)
            data["input"] = data["x"].to(device)
            data["output"] =data["y"].to(device)
            data["functions_cat"] =data["functions_cat"].to(device)

            optimizer.zero_grad()
            loss = F.mse_loss(model(data["input"]), data["output"])
            loss.backward()
            loss_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long))
 
            loss_cat.backward()
            optimizer.step()
        mlflow.log_metric(key="loss",value=loss,step=epoch)
        mlflow.log_metric(key="loss_cat",value=loss_cat,step=epoch)
    else:
        data = copy.deepcopy(data)
        data["input"] = data.to(device["input"])
        data["output"] = data.to(device["output"])
        data["functions_cat"] =data["functions_cat"].to(device)
        optimizer.zero_grad()
        F.mse_loss(model(data["input"]), data["output"]).backward()
        loss_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long))

        loss_cat.backward()
        optimizer.step()

def train_croosentr_cvae_cat(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()


    #return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    if data is None:
        for data in loader:
            data["output"] =data["y"].to(device)
            data["input"] = data["x"].to(device)
            mu, log_var = model.encode(data["input"])
            #print(len(model(data["input"])))

            kld_weight = param_dict['M_N']  # Account for the minibatch samples from the dataset
            recons_loss =F.mse_loss(model(data["input"]), torch.tensor(data["output"]))

            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            loss.mean().backward()
            #print('torch.tensor(data["functions_cat"],dtype=torch.long)',torch.tensor(data["functions_cat"],dtype=torch.long))
            loss_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long).to(device))
        
            loss_cat.backward()
            optimizer.step()
        mlflow.log_metric(key="recons_loss",value=recons_loss,step=epoch)
        mlflow.log_metric(key="kld_weight",value=kld_weight,step=epoch)
        mlflow.log_metric(key="kld_loss",value=kld_loss.mean(),step=epoch)
        mlflow.log_metric(key="loss_cat",value=loss_cat,step=epoch)
    else:
        data["output"] =data["y"].to(device)
        data["input"] = data["x"].to(device)
        mu, log_var = model.encode(data["input"])


        kld_weight = param_dict['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(model(data["input"]), data["input"])

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        loss.backward()
        loss_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long))
    
        loss_cat.backward()
        optimizer.step()

def train_croosentr_cvae_cat_concat_data(epoch,loader,optimizer,model,device,data = None,mlflow=None,param_dict={}):
    model.train()
    """
    
    Here I am concatinating the input and output and also I give the input as a separate entity
    
    """
    #return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    if data is None:
        for data in loader:
            data["data_1"] =data["y"].to(device)
            data["input"] = data["x"].to(device)
            mu, log_var = model.encode(data["input"])
            #print(len(model(data["input"])))

            kld_weight = param_dict['M_N']  # Account for the minibatch samples from the dataset
            recons_loss =F.mse_loss(model(data["input"],data["data_1"]), torch.tensor(data["input"]))

            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            loss.mean().backward()
            #print('torch.tensor(data["functions_cat"],dtype=torch.long)',torch.tensor(data["functions_cat"],dtype=torch.long))
            loss_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long).to(device))
        
            loss_cat.backward()
            optimizer.step()
        mlflow.log_metric(key="recons_loss",value=recons_loss,step=epoch)
        mlflow.log_metric(key="kld_weight",value=kld_weight,step=epoch)
        mlflow.log_metric(key="kld_loss",value=kld_loss.mean(),step=epoch)
        mlflow.log_metric(key="loss_cat",value=loss_cat,step=epoch)
    else:
        data["output"] =data["y"].to(device)
        data["input"] = data["x"].to(device)
        mu, log_var = model.encode(data["input"])


        kld_weight = param_dict['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(model(data["input"],data["data_1"]), data["input"])

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        loss.backward()
        loss_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long))
    
        loss_cat.backward()
        optimizer.step()