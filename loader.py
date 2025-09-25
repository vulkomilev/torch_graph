
import torch
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import re
import json
import numpy as np
from torch_geometric.data import Data

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
