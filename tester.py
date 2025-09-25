
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy


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
