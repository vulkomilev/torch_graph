
import torch
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import re
import json
import numpy as np
import copy
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import TensorDataset

class ArrayDatasetGraph(torch.utils.data.Dataset):

  def __init__(self, *args, **kwargs):
    super().__init__()
    # Your code
    self.x = kwargs['x']
    self.y = kwargs['y']
    self.dataset = {'x':[],'y':[]}
    for x , y,function in zip(kwargs['x'],kwargs['y']):
       self.dataset['x'].append(x)
       self.dataset['y'].append(y)

  def __getitem__(self, idx):
    return {'x':self.dataset['x'][idx],'y':self.dataset['y'][idx]} # In case you stored your data on a list called instances

  def __len__(self):
    return len(self.dataset['x'])

class ArrayDatasetAutoEncoder(torch.utils.data.Dataset):

  def __init__(self, *args, **kwargs):
    super().__init__()
    # Your code
    self.x = kwargs['x']
    self.y = kwargs['y']
    self.functions = kwargs['functions']
    self.dataset = {'x':[],'y':[],'functions':[],'functions_cat':[]}
    for x , y,function in zip(kwargs['x'],kwargs['y'],kwargs['functions']):
       self.dataset['x'].append(x)
       self.dataset['y'].append(y)
       self.dataset['functions'].append(function)
       local_name = ''
       for name in function:
            local_name += name[0]
       self.dataset['functions_cat'].append(kwargs['name_map'][local_name])

  def __getitem__(self, idx):
    return {'x':torch.tensor(self.dataset['x'][idx])
            ,'y':torch.tensor(self.dataset['y'][idx])
            ,'functions':self.dataset['functions'][idx]
            ,'functions_cat':torch.tensor(self.dataset['functions_cat'][idx])} # In case you stored your data on a list called instances

  def __len__(self):
    return len(self.dataset['x'])


def load_ARC_globalNode(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
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
                    for i in range(param_dict['hidden_node_count']):
                        m_i = np.append(m_i, 0.0)
     
                    m_i = torch.tensor(m_i)
                   # m_i = torch.cat([ m_i, torch.tensor(0.0)])
                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1]+param_dict['hidden_node_count'],1))

                    


                    m_i = m_i.expand((-1,1))
                    
                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = torch.tensor(m_o)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],1))
                    
                    for n in range(param_dict['hidden_node_count']):
                        for i in range(1,m_i_shape[0]*m_i_shape[1]):
                            connection_s.append(i)
                            connection_t.append(m_i_shape[0]*m_i_shape[1]+n)
                        for i in range(1,m_i_shape[0]*m_i_shape[1]):
                            connection_s.append(m_i_shape[0]*m_i_shape[1]+n)
                            connection_t.append(i)
                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr


def load_ARC_all_connected(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
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
                    for i in range(np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                        for j in range(i,np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                                connection_s.append(i)
                                connection_t.append(j)
                                

                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    
                    m_i = torch.tensor(m_i)
                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1],1))
                    m_i = m_i.expand((-1,1))
                    
                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = torch.tensor(m_o)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],1))

                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y = torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr

def load_ARC_Neighbor_Sampling(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
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
                    m_i = m_i.expand((-1,1))
                    
                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = torch.tensor(m_o)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],1))

                    return_arr  = NeighborLoader( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)),
                                input_nodes=torch.tensor([0, 1]),
                                num_neighbors=[2, 1],
                                batch_size=1,
                                replace=False,
                                shuffle=False,
                            )

    return return_arr

def load_ARC_separate_x_y(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
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

                    shape_i = list(np.array(examples['input']).shape)
                    shape_o = list(np.array(examples['output']).shape)
                    
                    #shape_i[0] += shape_o[0]
                    

                    connection_s = []
                    connection_t = []
                    for j in range(np.array(m_i).shape[1]):
                        for i in range(j,np.array(m_i).shape[0]*2*np.array(m_i).shape[1],np.array(m_i).shape[0]*2):
                            if i*np.array(m_i).shape[0]*2 < np.array(m_i).shape[0]*2*np.array(m_i).shape[1]:
                                connection_s.append(i)
                                connection_t.append(i*np.array(m_i).shape[0])
                                
                    for j in range(np.array(m_i).shape[1]):
                        for i in range(j,np.array(m_i).shape[0]*2-1):
                                connection_s.append(j*np.array(m_i).shape[0]+i)
                                connection_t.append(j*np.array(m_i).shape[0]+i+1)

                    
                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    
                    
                    #m_i = torch.tensor(np.zeros((shape_i)))
                    m_i = torch.tensor(m_i)
                    
                    #print(torch.zeros(shape_o[0]*shape_o[1]).shape)
                    #print(m_i.shape)
                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1],1))
                    
                    m_i = torch.cat((torch.zeros((shape_o[0]*shape_o[1],1)),m_i))
                    m_i = m_i.expand((-1,1))
                    
                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = torch.tensor(m_o)
                    
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],1))
                    m_o = torch.cat((m_o,torch.zeros((shape_i[0]*shape_i[1],1))))
                    #print('m_o',m_o.shape)
                    m_o = m_o.expand((-1,1))

                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y = torch.tensor(m_o,dtype=torch.float),
                                            edge_index = torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr = torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr


def load_ARC_GNN_with_symbolic_logic(size,num_features,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={"symbolic_function":None}):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    symbolic_function = param_dict["symbolic_function"]

    
    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
           with open(mypath+'/'+element,'r') as f:
            json_dict = json.load(f)
            for examples in json_dict['train']:

                    
                    #if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                    #    continue
                    rand_tens = torch.rand(param_dict['data_size_x'],param_dict['data_size_y'])
                    examples = {
                        'input' : rand_tens,
                        'output': symbolic_function(rand_tens)
                    }
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
                    m_i = np.concatenate((m_i,np.zeros(m_i_shape[0]*m_i_shape[1]*2)))
                    m_i = torch.tensor(m_i)
                    

                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1]+m_i_shape[0]*m_i_shape[1]*2,1))
                    m_i = m_i.expand((-1,num_features))

                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = np.concatenate((m_o,np.zeros(m_o_shape[0]*m_o_shape[1]*2)))
                    m_o = torch.tensor(m_o)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1]+m_o_shape[0]*m_o_shape[1]*2,1))
                    m_o = m_o.expand((-1,num_features))

                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr

def load_ARC_GNN_with_symbolic_logic_all_connected(size,num_features,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={"symbolic_function":None}):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    symbolic_function = param_dict["symbolic_function"]

    
    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
           with open(mypath+'/'+element,'r') as f:
            json_dict = json.load(f)
            for examples in json_dict['train']:

                    
                    if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                        continue
                    #rand_tens = torch.rand(param_dict['data_size_x'],param_dict['data_size_y'])
                    #examples = {
                    #    'input' : rand_tens,
                    #    'output': symbolic_function(rand_tens)
                    #}
                    m_i = examples['input']
                    connection_s = []
                    connection_t = []
                    for i in range(np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                        for j in range(i,np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                                connection_s.append(i)
                                connection_t.append(j)

                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    m_i = np.concatenate((m_i,np.zeros(m_i_shape[0]*m_i_shape[1]*2)))
                    m_i = torch.tensor(m_i)
                    

                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1]+m_i_shape[0]*m_i_shape[1]*2,1))
                    m_i = m_i.expand((-1,num_features))

                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = np.concatenate((m_o,np.zeros(m_o_shape[0]*m_o_shape[1]*2)))
                    m_o = torch.tensor(m_o)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1]+m_o_shape[0]*m_o_shape[1]*2,1))
                    m_o = m_o.expand((-1,num_features))

                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr

def load_ARC(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
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
                    m_i = m_i.expand((-1,1))
                    
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

def load_ARC_onehot(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
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

                    
                     #np.array(m_i).flatten()/10.0
                    
                    m_i = torch.tensor(m_i)
                    m_i = F.one_hot(m_i,num_classes=10)
                    m_i_shape = np.array(m_i).shape
                    #print('m_i_shape',m_i_shape)
                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1],10))

                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    #m_o = np.array(m_o).flatten()/10.0
                    
                    m_o = torch.tensor(m_o)
                    m_o = F.one_hot(m_o,num_classes=10)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],10))
    
                    return_arr.append( Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))

    return return_arr

def load_kaggle_ARC(size,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={}):
        return_list = []
        return_list_id = []
        #print("mypath",mypath)
        #print("json.load(mypath)",json.load(mypath))

        with open(mypath) as json_data:
            json_dict = json.load(json_data)
        print('len list(json_dict.keys())',len(list(json_dict.keys())))
        for id in list(json_dict.keys()):
            for examples in json_dict[id]['test']:


                    #if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                    #    continue
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
                    m_i = m_i.expand((-1,1))
                    if 'output' in list(examples.keys()):
                        m_o = examples['output']

                        m_o_shape = np.array(m_o).shape
                        m_o = np.array(m_o).flatten()/10.0
                        m_o = torch.tensor(m_o)
                        m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1],1))
                        #print('m_i_shape',m_i_shape)
                        #exit(0)
                    else:
                        m_o = torch.zeros((1,1))
                        m_o_shape = (1,1)
                    return_list.append(Data(x = torch.tensor(m_i,dtype=torch.float),
                                            y=torch.tensor(m_o,dtype=torch.float),
                                            edge_index=torch.tensor([connection_s,connection_t]),
                                            x_shape = m_i_shape,
                                            y_shape = m_o_shape,
                                            edge_attr= torch.tensor(np.random.random(np.array([connection_s,connection_t]).shape).transpose(),dtype=torch.float)))
                    return_list_id.append(id)
        return return_list,return_list_id

def load_ARC_AutoEncoder_syth_dataset(size,num_features,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={"symbolic_function":None}):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    symbolic_function = param_dict["symbolic_function"]
    functions_arr = []
    functions_names = []
    x_array = []
    y_array = []

    def encode_func_name(func_names):
       
        local_map = {}
        for i in range(len(func_names)):
            local_key = [0]*len(func_names)
            local_map[func_names[i]] = local_key
        return local_map

    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
           node_array = []
           with open(mypath+'/graphs/'+element[:-5]+".graph.json",'r') as f:
               json_dict = json.load(f)
               for node in json_dict['nodes']:
                   node_array.append(node[1])
               for node in node_array:
                   local_name = ''
                   for name in node_array:
                       local_name += name[0]
                   if local_name not in functions_names:
                       functions_names.append(local_name)
           functions_arr.append(node_array)

           with open(mypath+'/'+element,'r') as f:
            json_dict = json.load(f)
            for examples in json_dict['train']:

                    
                    if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                        continue
                    #rand_tens = torch.rand(param_dict['data_size_x'],param_dict['data_size_y'])
                    #examples = {
                    #    'input' : rand_tens,
                    #    'output': symbolic_function(rand_tens)
                    #}
                    m_i = examples['input']
                    connection_s = []
                    connection_t = []
                    for i in range(np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                        for j in range(i,np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                                connection_s.append(i)
                                connection_t.append(j)

                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    m_i = np.concatenate((m_i,np.zeros(m_i_shape[0]*m_i_shape[1]*2)))
                    m_i = torch.tensor(m_i,dtype=torch.float)
                    

                    m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1]+m_i_shape[0]*m_i_shape[1]*2,1))
                    m_i = m_i.expand((-1,num_features))

                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    m_o = np.concatenate((m_o,np.zeros(m_o_shape[0]*m_o_shape[1]*2)))
                    m_o = torch.tensor(m_o,dtype=torch.float)
                    m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1]+m_o_shape[0]*m_o_shape[1]*2,1))
                    m_o = m_o.expand((-1,num_features))
                    #print('m_i_shape',m_i_shape)
                    #exit(0)
                    pad_size_i = (0,0,0,abs(10000-m_i.shape[0]))#-m_i_shape[0]*m_i_shape[1]))
                    pad_size_o = (0,0,0,abs(10000-m_o.shape[0]))#-m_o_shape[0]*m_o_shape[1]))

                    m_i = F.pad(m_i, pad_size_i, "constant", 0)
                    m_o = F.pad(m_o, pad_size_o, "constant", 0)

                    x_array.append(m_i)
                    y_array.append(m_o)
    
                    
    #print('x_array',x_array[0])
    name_map = encode_func_name(functions_names)

    return ArrayDatasetAutoEncoder(x=x_array,y=y_array,functions=functions_arr,name_map=name_map)

def load_ARC_AutoEncoder_inputoutput_syth_dataset(size,num_features,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={"symbolic_function":None}):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    symbolic_function = param_dict["symbolic_function"]
    functions_arr = []
    functions_names = []
    x_array = []
    y_array = []

    def encode_func_name(func_names):
       
        local_map = {}
        for i in range(len(func_names)):
            local_key = [0]*len(func_names)
            local_map[func_names[i]] = local_key
        return local_map

    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
           node_array = []
           with open(mypath+'/graphs/'+element[:-5]+".graph.json",'r') as f:
               json_dict = json.load(f)
               for node in json_dict['nodes']:
                   node_array.append(node[1])
               for node in node_array:
                   local_name = ''
                   for name in node_array:
                       local_name += name[0]
                   if local_name not in functions_names:
                       functions_names.append(local_name)
           functions_arr.append(node_array)

           with open(mypath+'/'+element,'r') as f:
            json_dict = json.load(f)
            for examples in json_dict['train']:

                    
                    if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                        continue
                    #rand_tens = torch.rand(param_dict['data_size_x'],param_dict['data_size_y'])
                    #examples = {
                    #    'input' : rand_tens,
                    #    'output': symbolic_function(rand_tens)
                    #}
                    m_i = examples['input']
                    connection_s = []
                    connection_t = []
                    for i in range(np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                        for j in range(i,np.array(m_i).shape[0]*np.array(m_i).shape[1]):
                                connection_s.append(i)
                                connection_t.append(j)

                    m_i_shape = np.array(m_i).shape
                    m_i = np.array(m_i).flatten()/10.0
                    #m_i = np.concatenate((m_i,np.zeros(m_i_shape[0]*m_i_shape[1]*2)))
                    m_i = torch.tensor(m_i,dtype=torch.float)
                    

                    #m_i = m_i.reshape((m_i_shape[0]*m_i_shape[1]+m_i_shape[0]*m_i_shape[1]*2,1))
                    m_i = m_i.expand((-1,num_features))

                    m_o = examples['output']

                    m_o_shape = np.array(m_o).shape
                    m_o = np.array(m_o).flatten()/10.0
                    #m_o = np.concatenate((m_o,np.zeros(m_o_shape[0]*m_o_shape[1]*2)))
                    m_o = torch.tensor(m_o,dtype=torch.float)
                    #m_o = m_o.reshape((m_o_shape[0]*m_o_shape[1]+m_o_shape[0]*m_o_shape[1]*2,1))
                    m_o = m_o.expand((-1,num_features))
                    #print('m_i_shape',m_i_shape)
                    #exit(0)
                    pad_size_i = (0,0,0,abs(10000-m_i.shape[0]))#-m_i_shape[0]*m_i_shape[1]))
                    pad_size_o = (0,0,0,abs(10000-m_o.shape[0]))#-m_o_shape[0]*m_o_shape[1]))

                    m_i = F.pad(m_i, pad_size_i, "constant", 0)
                    m_o = F.pad(m_o, pad_size_o, "constant", 0)

                    x_array.append(m_i)
                    y_array.append(m_o)
    
                    
    #print('x_array',x_array[0])
    name_map = encode_func_name(functions_names)

    return ArrayDatasetAutoEncoder(x=x_array,y=y_array,functions=functions_arr,name_map=name_map)

def load_ARC_AutoEncoder_syth_concat_input_output_dataset(size,num_features,mypath = './datasets/abstraction-and-reasoning-challenge/training/',param_dict={"symbolic_function":None}):
    """
    
    Here I am concatinating the input and output and also I give the input as a separate entity
    
    """
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return_arr = []
    symbolic_function = param_dict["symbolic_function"]
    functions_arr = []
    functions_names = []
    x_array = []
    y_array = []

    def encode_func_name(func_names):
       
        local_map = {}
        for i in range(len(func_names)):
            local_key = [0]*len(func_names)
            local_map[func_names[i]] = local_key
        return local_map

    for element in onlyfiles[:size]:
        if re.findall(r'.*\.json$',element):
           node_array = []
           with open(mypath+'/graphs/'+element[:-5]+".graph.json",'r') as f:
               json_dict = json.load(f)
               for node in json_dict['nodes']:
                   node_array.append(node[1])
               for node in node_array:
                   local_name = ''
                   for name in node_array:
                       local_name += name[0]
                   if local_name not in functions_names:
                       functions_names.append(local_name)
           functions_arr.append(node_array)

           with open(mypath+'/'+element,'r') as f:
            json_dict = json.load(f)
            for examples in json_dict['train']:

                    
                    if np.array(examples['input']).flatten().shape !=  np.array(examples['output']).flatten().shape:
                        continue

                    m_i = examples['input']
                    m_o = examples['output']
                    concat_data = np.concat((m_i,m_o))

                    m_x = copy.deepcopy(m_i)

                    concat_data_shape = np.array(concat_data).shape
                    concat_data = np.array(concat_data).flatten()/10.0
                    concat_data = np.concatenate((concat_data,np.zeros(concat_data_shape[0]*concat_data_shape[1]*2)))
                    concat_data = torch.tensor(concat_data,dtype=torch.float)
                    

                    concat_data = concat_data.reshape((concat_data_shape[0]*concat_data_shape[1]+concat_data_shape[0]*concat_data_shape[1]*2,1))
                    concat_data = concat_data.expand((-1,num_features))


                    m_x_shape = np.array(m_x).shape
                    m_x = np.array(m_x).flatten()/10.0
                    m_x = np.concatenate((m_x,np.zeros(m_x_shape[0]*m_x_shape[1]*2)))
                    m_x = torch.tensor(m_x,dtype=torch.float)
                    m_x = m_x.reshape((m_x_shape[0]*m_x_shape[1]+m_x_shape[0]*m_x_shape[1]*2,1))
                    m_x = m_x.expand((-1,num_features))
                    #print('m_i_shape',m_i_shape)
                    #exit(0)
                    pad_size_i = (0,0,0,abs(10000-concat_data.shape[0]))#-m_i_shape[0]*m_i_shape[1]))
                    pad_size_o = (0,0,0,abs(10000-m_x.shape[0]))#-m_o_shape[0]*m_o_shape[1]))

                    concat_data = F.pad(concat_data, pad_size_i, "constant", 0)
                    m_x = F.pad(m_x, pad_size_o, "constant", 0)

                    x_array.append(concat_data)
                    y_array.append(m_x)
    
                    
    #print('x_array',x_array[0])
    name_map = encode_func_name(functions_names)

    return ArrayDatasetAutoEncoder(x=x_array,y=y_array,functions=functions_arr,name_map=name_map)