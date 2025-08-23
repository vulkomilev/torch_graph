
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER
import copy
from loader import load_ARC_onehot,load_ARC
from models import HotEncoding,HotEncodingForLoopForward,CNN,CNNFor
from trainer import train_cross_entropy,train_mse
from tester import test,test_one_hot
import mlflow
from mlflow.models import infer_signature
import optuna

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

EPOCHS =100
TRAIN_EPOCHS = 1
N_TRIALS = 100
IMAGE_SHOW_TRESHOLD = 1
total_metric_min = 1.0
local_device = 'cuda:0'
EXPERIMENT_NAME = 'test'
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment_id is None:
    # If the experiment does not exist, create it
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
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
train_dataset = load_ARC(300,mypath = './datasets/abstraction-and-reasoning-challenge/training/')
test_dataset = load_ARC(300,mypath = './datasets/abstraction-and-reasoning-challenge/test/')

#print('train_dataset',train_dataset[0].x.shape)
#print('test_dataset',test_dataset[0].x.shape)
#train_dataset = train_dataset[:1]
#test_dataset = test_dataset[:1]
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
def pipeline(trial=None):
    global mlflow
    total_metric_min = 1.0
    if trial is None:
        lr = 0.001
    else:
        lr = trial.suggest_float("lr", 0.0002, 0.005, log=True)
        for_cycles = trial.suggest_int("for_cycles", 0, 300)
    mlflow.log_param(key='lr',value=lr)
    mlflow.log_param(key='for_cycles',value=for_cycles)
    model = CNNFor(for_cycles=for_cycles).to(local_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(0, EPOCHS):
     train_mse(epoch,train_loader,optimizer,model,local_device,None,mlflow)
    torch.save(model.state_dict(), "./model")
    checkpoint = torch.load("./model", weights_only=False)
    model.load_state_dict(checkpoint)
    test_acc,total_metric_min_result = test(epoch,model,test_loader,TRAIN_EPOCHS,IMAGE_SHOW_TRESHOLD,train_mse,local_device,train_dataset,optimizer,total_metric_min,mlflow)
    print(f'Test: {test_acc:.4f}, Min: {total_metric_min_result:.4f}')
    return total_metric_min_result

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


