import matplotlib.pyplot as plt
import torch
import numpy as np
import copy

import torch.nn.functional as F
def one_hot_decode(x):
    x_new = np.zeros(x.shape[:2])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #print('np.where(x[i][j] == 1)',np.where(x[i][j] == 1)[0][0])
            x_new[i][j] =  int(np.where(x[i][j] == 1)[0][0])#x[i][j].index(1)

    return x_new

def test_globalNode(epoch,model,loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_func,device,train_dataset,optimizer,total_metric_min,mlflow,param_dict={}):
    
    model.eval()
    correct = 0

    for data in loader:
        for epoch_train in range(1, TRAIN_EPOCHS):
            train_func(epoch_train,train_dataset,optimizer,model,device,None)
        data = data.to(device)
        data_1 = copy.deepcopy(data)
        pred = model(data_1).flatten()
        cos_sim = torch.dot(pred[:-1*param_dict['hidden_node_count']], data.y.flatten()).flatten()/(torch.norm(pred)*torch.norm(data.y.flatten()))
     
        total_metric = 1-cos_sim.cpu().detach().numpy().tolist()[0]
        correct += total_metric 
        mlflow.log_metrics({"total_metric": total_metric,"total_metric_min": total_metric_min}  )
        if total_metric_min > total_metric:
            total_metric_min = total_metric
        # if total_metric <= IMAGE_SHOW_TRESHOLD  or epoch > 20:
        #     figure, axis = plt.subplots(1, 3)
        #     axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
        #     axis[ 0].set_title("Predicted")

        #     axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
        #     axis[ 1].set_title("True")
            
        #     axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
        #     axis[ 2].set_title("Input")
        #     plt.show()

    print('total_metric_min',total_metric_min)
    return correct / len(loader),total_metric_min

def test(epoch,model,loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_func,device,train_dataset,optimizer,total_metric_min,mlflow,param_dict={}):
    
    model.eval()
    correct = 0
    for data in loader:
        for epoch_train in range(1, TRAIN_EPOCHS):
            train_func(epoch_train,train_dataset,optimizer,model,device,None)
        data = data.to(device)
       
        data_1 = copy.deepcopy(data)
        pred = model(data_1).flatten()
        #print("data",data_1.x_shape)
        #print("data",data_1.y_shape)
        #print("pred",pred.reshape((data_1.x_shape[0][0],data_1.x_shape[0][1],1)).shape)
        cos_sim = torch.dot(pred, data.y.flatten()).flatten()/(torch.norm(pred)*torch.norm(data.y.flatten()))
     
        total_metric = 1-cos_sim.cpu().detach().numpy().tolist()[0]
        correct += total_metric 
        mlflow.log_metrics({"total_metric": total_metric,"total_metric_min": total_metric_min}  )
        if total_metric_min > total_metric:
            total_metric_min = total_metric
        # if total_metric <= IMAGE_SHOW_TRESHOLD  or epoch > 20:
        #     figure, axis = plt.subplots(1, 3)
        #     axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
        #     axis[ 0].set_title("Predicted")

        #     axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
        #     axis[ 1].set_title("True")
            
        #     axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
        #     axis[ 2].set_title("Input")
        #     plt.show()

    print('total_metric_min',total_metric_min)
    return correct / len(loader),total_metric_min

def test_one_hot(epoch,model,loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_func,device,train_dataset,optimizer,total_metric_min,mlflow,param_dict={}):
    model.eval()
    correct = 0

    for data in loader:
        for epoch_train in range(1, TRAIN_EPOCHS):
           train_func(epoch_train,train_dataset,optimizer,model,device,None)
        data = data.to(device)
        data_1 = copy.deepcopy(data)
        pred = model(data_1).flatten()
        cos_sim = torch.dot(pred, data.y.flatten()).flatten()/(torch.norm(pred)*torch.norm(data.y.flatten()))
        total_metric = 1-cos_sim.cpu().detach().numpy().tolist()[0]
    
        mlflow.log_metric("total_metric", total_metric)
        mlflow.log_metric("total_metric_min", total_metric_min)
        correct += total_metric 

        if total_metric_min > total_metric:
            total_metric_min = total_metric
        # if total_metric <= IMAGE_SHOW_TRESHOLD  or epoch > 20:
        #     figure, axis = plt.subplots(1, 3)
        #     data_y = one_hot_decode(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1],10)))
        #     axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
        #     axis[ 0].set_title("Predicted")

        #     axis[ 1].imshow(data_y/10.0)
        #     axis[ 1].set_title("True")
            
        #     axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
        #     axis[ 2].set_title("Input")
        #     plt.show()
    print('total_metric_min',total_metric_min)

    return correct / len(loader),total_metric_min

def test_autoencoder(epoch,model,loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_func,device,train_dataset,optimizer,total_metric_min,mlflow,param_dict={}):
    
    model.eval()
    correct = 0
    for data in loader:
        for epoch_train in range(1, TRAIN_EPOCHS):
            train_func(epoch_train,train_dataset,optimizer,model,device,None)
        data = data.to(device)
       
        data["input"] = data["input"].to(device)
        data["output"] = data["output"].to(device)
        pred = model(data["input"]).flatten()
        #print("data",data_1.x_shape)
        #print("data",data_1.y_shape)
        #print("pred",pred.reshape((data_1.x_shape[0][0],data_1.x_shape[0][1],1)).shape)
        cos_sim = torch.dot(pred, data["output"].flatten()).flatten()/(torch.norm(pred)*torch.norm(data["output"].flatten()))
     
        total_metric = 1-cos_sim.cpu().detach().numpy().tolist()[0]
        correct += total_metric 
        mlflow.log_metrics({"total_metric": total_metric,"total_metric_min": total_metric_min}  )
        if total_metric_min > total_metric:
            total_metric_min = total_metric
        # if total_metric <= IMAGE_SHOW_TRESHOLD  or epoch > 20:
        #     figure, axis = plt.subplots(1, 3)
        #     axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
        #     axis[ 0].set_title("Predicted")

        #     axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
        #     axis[ 1].set_title("True")
            
        #     axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
        #     axis[ 2].set_title("Input")
        #     plt.show()

    print('total_metric_min',total_metric_min)
    return correct / len(loader),total_metric_min

def test_autoencoder_func_cat(epoch,model,loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_func,device,train_dataset,optimizer,total_metric_min,mlflow,param_dict={}):
    
    model.eval()
    correct = 0
    correlate_dataset = {}
    cat_pred_dataset = {}
    for data in loader:
        for epoch_train in range(1, TRAIN_EPOCHS):
            train_func(epoch_train,train_dataset,optimizer,model,device,None)
        
        data = copy.deepcopy(data)
        data["input"] = data["x"].to(device)
        data["output"] = data["y"].to(device)
        local_functions = data["functions"]
        local_name = ""

        for element in local_functions[0]:
            
                local_name+=element
        if local_name not in list(correlate_dataset.keys()):
            correlate_dataset[local_name] = []
        if local_name not in list(cat_pred_dataset.keys()):
            cat_pred_dataset[local_name] = [] 

  
        cos_sim_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long).to(device)).cpu().detach()#/(torch.norm(pred_cat)*torch.norm( torch.tensor(data['functions_cat'],dtype=torch.float)))
        cat_pred_dataset[local_name].append(cos_sim_cat)
        #print("data",data_1.x_shape)
        #print("data",data_1.y_shape)
        #print("pred",pred.reshape((data_1.x_shape[0][0],data_1.x_shape[0][1],1)).shape)

        # if total_metric <= IMAGE_SHOW_TRESHOLD  or epoch > 20:
        #     figure, axis = plt.subplots(1, 3)
        #     axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
        #     axis[ 0].set_title("Predicted")

        #     axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
        #     axis[ 1].set_title("True")
            
        #     axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
        #     axis[ 2].set_title("Input")
        #     plt.show()
    for key in list(correlate_dataset.keys()):
        mlflow.log_metrics({key+'_catcos':np.mean(cat_pred_dataset[key])}  )
    print('total_metric_min',total_metric_min)
    return correct / len(loader),total_metric_min

def test_autoencoder_func_cat_concat_data(epoch,model,loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_func,device,train_dataset,optimizer,total_metric_min,mlflow,param_dict={}):
    """
    
    Here I am concatinating the input and output and also I give the input as a separate entity
    
    """
    correct = 0
    correlate_dataset = {}
    cat_pred_dataset = {}
    for data in loader:
        for epoch_train in range(1, TRAIN_EPOCHS):
            train_func(epoch_train,train_dataset,optimizer,model,device,None)
        
        data = copy.deepcopy(data)
        data["data_1"] =data["y"].to(device)
        data["input"] = data["x"].to(device)
        local_functions = data["functions"]
        local_name = ""

        for element in local_functions[0]:
            
                local_name+=element
        if local_name not in list(correlate_dataset.keys()):
            correlate_dataset[local_name] = []
        if local_name not in list(cat_pred_dataset.keys()):
            cat_pred_dataset[local_name] = [] 

  
        cos_sim_cat = F.cross_entropy(model.categ(data["input"]),torch.tensor(data["functions_cat"],dtype=torch.long).to(device)).cpu().detach()#/(torch.norm(pred_cat)*torch.norm( torch.tensor(data['functions_cat'],dtype=torch.float)))
        cat_pred_dataset[local_name].append(cos_sim_cat)
        #print("data",data_1.x_shape)
        #print("data",data_1.y_shape)
        #print("pred",pred.reshape((data_1.x_shape[0][0],data_1.x_shape[0][1],1)).shape)

        # if total_metric <= IMAGE_SHOW_TRESHOLD  or epoch > 20:
        #     figure, axis = plt.subplots(1, 3)
        #     axis[ 0].imshow(np.rot90(pred.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])),k=0))
        #     axis[ 0].set_title("Predicted")

        #     axis[ 1].imshow(data.y.cpu().detach().numpy()[:data.y_shape[0][0]*data.y_shape[0][1]].reshape((data.y_shape[0][0],data.y_shape[0][1])))
        #     axis[ 1].set_title("True")
            
        #     axis[ 2].imshow(data.x[:,:1].cpu().detach().numpy()[:data.x_shape[0][0]*data.x_shape[0][1]].reshape((data.x_shape[0][0],data.x_shape[0][1])))
        #     axis[ 2].set_title("Input")
        #     plt.show()
    for key in list(correlate_dataset.keys()):
        mlflow.log_metrics({key+'_catcos':np.mean(cat_pred_dataset[key])}  )
    print('total_metric_min',total_metric_min)
    return correct / len(loader),total_metric_min
