
import matplotlib.pyplot as plt
import torch
import numpy as np
import inspect
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DataLoaderDefault
from torch_geometric.typing import WITH_TORCH_CLUSTER
import copy
from loader import load_ARC_onehot,load_ARC,load_ARC_all_connected,load_ARC_globalNode,load_ARC_Neighbor_Sampling,load_ARC_separate_x_y,load_ARC_GNN_with_symbolic_logic,\
load_ARC_GNN_with_symbolic_logic_all_connected,load_ARC_AutoEncoder_syth_dataset,load_ARC_AutoEncoder_syth_concat_input_output_dataset
from models import HotEncoding,HotEncodingForLoopForward,CNN,CNNFor,CNN_Symbolic_Parallel,CNN_with_symbolic_logic,CNN_with_symbolic_logic_attention,\
EdgeConv_with_symbolic_logic_attention,AutoEncoder,ConvVariationalAutoEncoder,ConvVariationalAutoEncoderConcatData
from trainer import train_cross_entropy,train_mse,train_mse_globalNode,train_mse_Neighbor_Sampling,train_mse_autoencoder,train_crossentr_autoencoder_cat,\
train_croosentr_cvae_cat,train_croosentr_cvae_cat_concat_data


from tester import test,test_one_hot,test_globalNode,test_autoencoder,test_autoencoder_func_cat,test_autoencoder_func_cat_concat_data
import mlflow
from mlflow.models import infer_signature
import optuna
import base64
from typing import Callable, Optional, Union


if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

EPOCHS = 1000
TRAIN_EPOCHS = 0
N_TRIALS = 100
IMAGE_SHOW_TRESHOLD = 1
total_metric_min = 1.0
local_device = 'cuda:0'
EXPERIMENT_NAME = 'load_ARC_GNN_with_symbolic_logic'
description = 'I am trying to add symbolic logic glued to the  graph neural network and I want to see if this symbolic logic can be used to solve the problem.' \
'For the task the symbolic logic used to generate the probelms and the symbolic logic glued to neural network will be the same.Testing with decomposed functions.'

def symbolic_function_rotate(tensor):
    tensor_shape = tensor.shape
    new_tensor = tensor.clone()
    new_tensor = torch.rot90(new_tensor)
    new_tensor = torch.reshape(new_tensor,tensor_shape)
    return new_tensor

def symbolic_function_asign_1(tensor):
            new_tensor = tensor.clone()
            local_dim = new_tensor.dim()
            if local_dim == 3:
                if tensor[0][0][0] < 0.2:
                        new_tensor[0][0][0] = 1
            else:
                if tensor[0][0] < 0.2:
                        new_tensor[0][0] = 1
            return new_tensor

def symbolic_function_asign_2(tensor):
            new_tensor = tensor.clone()
            local_dim = new_tensor.dim()
            if local_dim == 3:
                if tensor[0][0][0] < 0.12:
                        new_tensor[0][0][0] = 1
            else:
                if tensor[0][0] < 0.12:
                        new_tensor[0][0] = 1
            return new_tensor

def symbolic_function(tensor):
        tensor_shape = tensor.shape
        tensor = torch.reshape(tensor, (9, 9))
        new_tensor = tensor.clone()
        new_tensor = torch.rot90(new_tensor)
        for i in range(1,9):

            for j in range(9):
                if i > 5:
                      if tensor[j][5] < 0.12:
                       new_tensor[j][i] = 1 
                else:
                    if tensor[j][0] < 0.2:
                       new_tensor[j][i] = 1
        new_tensor = torch.reshape(new_tensor,tensor_shape)
        return new_tensor

def symbolic_function_fake(tensor):
        tensor_shape = tensor.shape
        tensor = torch.reshape(tensor, (9, 9))
        new_tensor = tensor.clone()
        new_tensor = torch.rot90(new_tensor)
        for i in range(1,2):

            for j in range(2):
                if i > 5:
                      if tensor[j][5] < 0.12:
                       new_tensor[j][i] = 1 
                else:
                    if tensor[j][0] < 0.5:
                       new_tensor[j][i] = 1
        new_tensor = torch.reshape(new_tensor,tensor_shape)
        return new_tensor


def symbolic_function_fake_push(tensor):
        tensor_shape = tensor.shape
        tensor = torch.reshape(tensor, (9, 9))
        new_tensor = tensor.clone()
        new_tensor = torch.rot90(new_tensor)
        new_tensor = torch.roll(new_tensor, shifts=-1, dims=1)   
        return new_tensor

symbolic_function = symbolic_function
param_dict = {"hidden_node_count":5,
              "symbolic_function":symbolic_function,
              "symbolic_function_fake":symbolic_function_fake_push,
              "symbolic_function_asign_2":symbolic_function_asign_2,
              "symbolic_function_asign_1":symbolic_function_asign_1,
              "symbolic_function_rotate":symbolic_function_rotate,
              "data_size":9*9,
              "data_size_x":9,
              "data_size_y":9,
              "M_N":0.1,
              "latent_dim":9,
              "hidden_dims":9
              }

local_loader = load_ARC_AutoEncoder_syth_concat_input_output_dataset
local_model = ConvVariationalAutoEncoderConcatData
local_trainer = train_croosentr_cvae_cat_concat_data
local_tester = test_autoencoder_func_cat_concat_data

num_features = 1 #15**2
num_target = 1 #15**2


experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment_id is None:
    # If the experiment does not exist, create it
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME,   tags={'mlflow.note.content':description})
else:
    # If the experiment exists, get its ID
    experiment_id = experiment_id.experiment_id

compability_dicts = {
    "model_":{
    "testers":[],
    "loaders":[]}
}



def champion_callback(study, frozen_trial):
  """
  Logging callback that will report when a new trial iteration improves upon existing
  best trial values.

  Note: This callback is not intended for use in distributed computing systems such as Spark
  or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
  workers or agents.
  The race conditions with file system state management for distributed trials will render
  inconsistent values with this callback.
  """

  winner = study.user_attrs.get("winner", None)

  if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
train_dataset = local_loader(300,num_features,mypath = './dataset_2function/',param_dict=param_dict)
test_dataset = local_loader(300,num_features,mypath = './dataset_2function/',param_dict=param_dict)
def collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    elem = batch[0]
    out = None
    # if elem.is_nested:
    #     raise RuntimeError(
    #         "Batches of nested tensors are not currently supported by the default collate_fn; "
    #         "please provide a custom collate_fn to handle them appropriately."
    #     )
    # if elem.layout in {
    #     torch.sparse_coo,
    #     torch.sparse_csr,
    #     torch.sparse_bsr,
    #     torch.sparse_csc,
    #     torch.sparse_bsc,
    # }:
    #     raise RuntimeError(
    #         "Batches of sparse tensors are not currently supported by the default collate_fn; "
    #         "please provide a custom collate_fn to handle them appropriately."
    #     )
    # if torch.utils.data.get_worker_info() is not None:
    #     # If we're in a background process, concatenate directly into a
    #     # shared memory tensor to avoid an extra copy
    #     numel = sum(x.numel() for x in batch)
    #     storage = elem._typed_storage()._new_shared(numel, device=elem.device)
    #     out = elem.new(storage).resize_(len(batch), *list(elem.size()))

    staked_batch = {}
    for key in list(batch[0].keys()):
        if key in ['x','y','functions_cat']:
         staked_batch[key] = torch.tensor(batch[0][key])
         staked_batch[key] = staked_batch[key].unsqueeze(0)
        else:
         staked_batch[key] = []
    for key in list(batch[0].keys()):
         for i in range(len(batch)):
             if key in ['x','y','functions_cat']:
                local_tensor = torch.tensor(batch[i][key])
                local_tensor = local_tensor.unsqueeze(0)
                staked_batch[key] = torch.cat((staked_batch[key],local_tensor))
             else:
                staked_batch[key].append(batch[i][key])
    #exit(0)
    return staked_batch

train_loader = DataLoaderDefault(train_dataset, batch_size=10, shuffle=False,collate_fn=collate_tensor_fn)
test_loader = DataLoaderDefault(test_dataset, batch_size=1, shuffle=False,collate_fn=collate_tensor_fn)
def pipeline(trial=None):
    global mlflow
    total_metric_min = 1.0
    if trial is None:
        lr = 0.001
    else:

        #use lr between 0.0002 and 0.005
        lr = trial.suggest_float("lr", 0.0002, 0.005, log=True)
        M_N = trial.suggest_float("M_N", 0.00001, 1, log=True)
        latent_dim = trial.suggest_int("latent_dim", 1, 150, log=True)
        hidden_dims = trial.suggest_int("hidden_dims", 1, 150, log=True)
        
        param_dict['M_N'] = M_N 
        param_dict['latent_dim'] = hidden_dims 
        param_dict['hidden_dims'] = hidden_dims 
    mlflow.sklearn.log_model(
        sk_model=local_loader,
        name="elasticnet",
        params={
            "local_loader": local_loader.__name__,
            "local_model": local_model.__name__,
            "local_trainer": local_trainer.__name__,
            "local_tester": local_tester.__name__,
        },)
    mlflow.log_param(key='lr',value=lr)
    model = local_model(num_features=num_features,num_target=num_target,param_dict=param_dict).to(local_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(0, EPOCHS):
      #pass

     local_trainer(epoch,train_loader,optimizer,model,local_device,None,mlflow,param_dict=param_dict)
    torch.save(model.state_dict(), "./model")
    checkpoint = torch.load("./model", weights_only=False)
    model.load_state_dict(checkpoint)
    test_acc,total_metric_min_result = local_tester(epoch,model,test_loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,local_trainer,local_device,train_dataset,optimizer,total_metric_min,mlflow,param_dict=param_dict)
    print(f'Test: {test_acc:.4f}, Min: {total_metric_min_result:.4f}')
    return total_metric_min_result

def test(trial=None):
    if trial is None:
        lr = 0.001
        for_cycles = 1
    else:
        #use lr between 0.0002 and 0.005
        lr = trial.suggest_float("lr", 0.0002, 0.005, log=True)
        for_cycles = trial.suggest_int("for_cycles", 0, 10)
    param_dict['for_cycles'] = for_cycles
    model = local_model(param_dict=param_dict).to(local_device)
    checkpoint = torch.load("./model", weights_only=False)
    model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(0, EPOCHS):

        test_acc,total_metric_min_result = local_tester(epoch,model,test_loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,local_trainer,local_device,train_dataset,optimizer,total_metric_min,mlflow,param_dict=param_dict)
        print(f'Test: {test_acc:.4f}, Min: {total_metric_min_result:.4f}')

#mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
parent_run = ''

def objective(trial=None):

    with mlflow.start_run( tags= {'parent_run':parent_run},experiment_id=experiment_id,nested=True):
        #mlflow.autolog()
        _ =  pipeline(trial)
    return _

with mlflow.start_run( experiment_id=experiment_id,nested=False) as run:
  # Initialize the Optuna study
  study = optuna.create_study(direction="minimize")
  parent_run = run.to_dictionary()['info']['run_name']
  print("tags.parent_run like '"+parent_run+"'")
  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
  study.optimize(objective, n_trials=N_TRIALS, callbacks=[champion_callback])


