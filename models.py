
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential,Conv1d,MultiheadAttention,ConvTranspose1d,Sigmoid,Module


import numpy as np

from torch_geometric.nn import (
    NNConv,
    GraphConv,
    EdgeConv,
    SimpleConv,
    TransformerConv
)

from torch_geometric.nn.attention import (
    PerformerAttention
)
class ConvVariationalAutoEncoderConcatData(Module):
    """
    
    Here I am concatinating the input and output and also I give the input as a separate entity
    
    """
    def __init__(self,num_features=None,num_target=None,param_dict=None):
        super(ConvVariationalAutoEncoderConcatData, self).__init__()
        self.latent_dim  = param_dict['latent_dim']
        self.hidden_dims  = param_dict['hidden_dims']
        self.encoder = Sequential(
            Linear(1, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 36),
            ReLU(),
            Linear(36, 18),
            ReLU(),
            Linear(18, self.hidden_dims)
        )
        self.decoder = Sequential(
            Linear( 1+self.hidden_dims, 18),
            ReLU(),
            Linear(18, 36),
            ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128,1),
            Sigmoid()
        )
        self.categorizer = Sequential(
            Linear(self.hidden_dims, 18),
            ReLU(),
            Linear(18, 36),
            ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128,2),
            Sigmoid()
        )
        self.fc_mu = Linear(self.hidden_dims, self.latent_dim)
        self.fc_var =Linear(self.hidden_dims, self.latent_dim)
        
    def encode(self,x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
    
    def categ(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.categorizer(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x,data_1):
        #encoded = self.encode(x)

        #decoded = self.decoder(encoded)


        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        #return  [self.decode(z), input, mu, log_var]
        return  self.decoder(torch.concat([data_1,z],dim=2))#[self.decoder(z), input, mu, log_var]
    
class ConvVariationalAutoEncoder(Module):
    def __init__(self,num_features=None,num_target=None,param_dict=None):
        super(ConvVariationalAutoEncoder, self).__init__()
        self.latent_dim  = param_dict['latent_dim']
        self.hidden_dims  = param_dict['hidden_dims']
        self.encoder = Sequential(
            Linear(1, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 36),
            ReLU(),
            Linear(36, 18),
            ReLU(),
            Linear(18, self.hidden_dims)
        )
        self.decoder = Sequential(
            Linear( 1+self.hidden_dims, 18),
            ReLU(),
            Linear(18, 36),
            ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128,1),
            Sigmoid()
        )
        self.categorizer = Sequential(
            Linear(self.hidden_dims, 18),
            ReLU(),
            Linear(18, 36),
            ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128,2),
            Sigmoid()
        )
        self.fc_mu = Linear(self.hidden_dims, self.latent_dim)
        self.fc_var =Linear(self.hidden_dims, self.latent_dim)
        
    def encode(self,x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
    
    def categ(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.categorizer(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        #encoded = self.encode(x)

        #decoded = self.decoder(encoded)


        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        #return  [self.decode(z), input, mu, log_var]
        return  self.decoder(torch.concat([x,z],dim=2))#[self.decoder(z), input, mu, log_var]
    
class AutoEncoder(Module):
    def __init__(self,num_features=None,num_target=None,param_dict=None):
        super(AutoEncoder, self).__init__()

        self.encoder = Sequential(
            Linear(1, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 36),
            ReLU(),
            Linear(36, 18),
            ReLU(),
            Linear(18, 9)
        )
        self.decoder = Sequential(
            Linear(9, 18),
            ReLU(),
            Linear(18, 36),
            ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128,1),
            Sigmoid()
        )
        self.categorizer = Sequential(
            Linear(9, 18),
            ReLU(),
            Linear(18, 36),
            ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128,2),
            Sigmoid()
        )
    def encode(self,x):
        return self.encoder(x)
    
    def categ(self,x):
        return self.categorizer(self.encoder(x))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class CNNFor(torch.nn.Module):
    def __init__(self,for_cycles=1,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        self.for_cycles=for_cycles
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')


        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        for i in range(self.for_cycles):
         data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))

class CNNGlobalNode(torch.nn.Module):
    def __init__(self,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')


        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))

        data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
 
        data.x = F.elu(self.cnn(torch.reshape(data.x, (1,-1))))
        data.x = F.elu(self.cnnT(torch.reshape(data.x, (1,-1))))
        data.x = torch.reshape(data.x, (-1,1))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))

class CNN_Symbolic_Parallel (torch.nn.Module):
    def __init__(self,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')


        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)
        self.fc3 = torch.nn.Linear(num_target*2, num_target)
   
    def symbolic_function(self,data):
        for i in range(data.shape[0]):
            if data[i] > 0:
                data[i] = 1.0
        return data

    def forward(self, data):
 
        data_new = self.symbolic_function(data.x.clone())
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))

        data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
 
        data.x = F.elu(self.cnn(torch.reshape(data.x, (1,-1))))
        data.x = F.elu(self.cnnT(torch.reshape(data.x, (1,-1))))
        data.x = torch.reshape(data.x, (-1,1))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.fc2(x))
        x = torch.concat([x,data_new],dim=1)
        x = F.elu(self.fc3(x))
        
        return x

class EdgeConv_with_symbolic_logic_attention(torch.nn.Module):
    def __init__(self,for_cycles=1,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={"symbolic_function":None,"data_size":None}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        self.for_cycles=param_dict["for_cycles"]
        self.symbolic_function = param_dict["symbolic_function"]
        self.symbolic_function_1 = param_dict["symbolic_function_asign_2"] 
        self.symbolic_function_2 = param_dict["symbolic_function_asign_1"]  
        self.symbolic_function_3 = param_dict["symbolic_function_rotate"]    
        self.data_size = param_dict["data_size"]
        nn1 = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = PerformerAttention(num_features,1)
        self.conv_m = PerformerAttention(num_features,1)
        self.multihead_attn = MultiheadAttention(num_features, num_features)

        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = PerformerAttention(num_features,1)


        self.fc1 = torch.nn.Linear(num_features, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
        data.x = data.x[None,:,:]
        data.x = F.elu(self.conv1(data.x))
        for i in range(self.for_cycles):
         pass
         #data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         #data.x[:,-self.data_size:] = self.symbolic_function_1(data.x[:,-2*self.data_size:-self.data_size])
         #data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         #data.x[:,-self.data_size:] = self.symbolic_function_2(data.x[:,-2*self.data_size:-self.data_size])
         #data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         #data.x[:,-self.data_size:] = self.symbolic_function_3(data.x[:,-2*self.data_size:-self.data_size])

        data.x = F.elu(self.conv2(data.x))
        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))
    
class CNN_with_symbolic_logic_attention_kaggle(torch.nn.Module):
    def __init__(self,for_cycles=1,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={"symbolic_function":None,"data_size":None}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        self.for_cycles=param_dict["for_cycles"]
        self.symbolic_function = param_dict["symbolic_function"]
        self.symbolic_function_1 = param_dict["symbolic_function_asign_2"] 
        self.symbolic_function_2 = param_dict["symbolic_function_asign_1"]  
        self.symbolic_function_3 = param_dict["symbolic_function_rotate"]    
        self.data_size = param_dict["data_size"]
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')
        self.multihead_attn = MultiheadAttention(num_features, 1)

        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        for i in range(self.for_cycles):
         data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         data.x[-self.data_size:] = self.symbolic_function_1(data.x[-2*self.data_size:-self.data_size])
         data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         data.x[-self.data_size:] = self.symbolic_function_2(data.x[-2*self.data_size:-self.data_size])
         data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         data.x[-self.data_size:] = self.symbolic_function_3(data.x[-2*self.data_size:-self.data_size])

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))
    
class CNN_with_symbolic_logic_attention(torch.nn.Module):
    def __init__(self,for_cycles=1,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={"symbolic_function":None,"data_size":None}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        self.for_cycles=param_dict["for_cycles"]
        self.symbolic_function = param_dict["symbolic_function"]
        self.symbolic_function_1 = param_dict["symbolic_function_asign_2"] 
        self.symbolic_function_2 = param_dict["symbolic_function_asign_1"]  
        self.symbolic_function_3 = param_dict["symbolic_function_rotate"]    
        self.data_size = param_dict["data_size"]
        nn1 = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')
        self.multihead_attn = MultiheadAttention(num_features, 1)

        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        for i in range(self.for_cycles):
         data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         data.x[-self.data_size:] = self.symbolic_function_1(data.x[-2*self.data_size:-self.data_size])
         data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         data.x[-self.data_size:] = self.symbolic_function_2(data.x[-2*self.data_size:-self.data_size])
         data.x, _ = self.multihead_attn(data.x, data.x, data.x)
         data.x[-self.data_size:] = self.symbolic_function_3(data.x[-2*self.data_size:-self.data_size])

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))

class CNN_with_symbolic_logic(torch.nn.Module):
    def __init__(self,for_cycles=1,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={"symbolic_function":None,"data_size":None}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        self.for_cycles=param_dict["for_cycles"]
        self.symbolic_function = param_dict["symbolic_function"]
        self.symbolic_function_1 = param_dict["symbolic_function_asign_2"] 
        self.symbolic_function_2 = param_dict["symbolic_function_asign_1"]  
        self.symbolic_function_3 = param_dict["symbolic_function_rotate"]    
        self.data_size = param_dict["data_size"]
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')


        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
 

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        for i in range(self.for_cycles):
         data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
         data.x[-self.data_size:] = self.symbolic_function_1(data.x[-2*self.data_size:-self.data_size])
         data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
         data.x[-self.data_size:] = self.symbolic_function_2(data.x[-2*self.data_size:-self.data_size])
         data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
         data.x[-self.data_size:] = self.symbolic_function_3(data.x[-2*self.data_size:-self.data_size])

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))

class CNN(torch.nn.Module):
    def __init__(self,num_features = 1,num_target = 1,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1),
        )
        nn3 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 1 * 64),
        )
        self.conv1 = NNConv(num_features, 1, nn1, aggr='mean')
        self.conv_m = NNConv(1, 1, nn2, aggr='mean')


        self.cnn = Conv1d(1,1,kernel_size=8)
        self.cnnT = ConvTranspose1d(1,1,kernel_size=8)
        self.conv2 = NNConv(1, 64, nn3, aggr='mean')


        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))

        data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
 
        data.x = F.elu(self.cnn(torch.reshape(data.x, (1,-1))))
        data.x = F.elu(self.cnnT(torch.reshape(data.x, (1,-1))))
        data.x = torch.reshape(data.x, (-1,1))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training)
        return F.elu(self.fc2(x))

class Attention(torch.nn.Module):
    def __init__(self,num_features = 10,num_target = 10,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        
        nn_start = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25,num_features),
        )
        self.nn_middle = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25, num_features),
        )
        nn_end = Sequential(
            Linear(num_features, 25),
            ReLU(),
            Linear(25,1* 64),
        )

        self.conv1 = NNConv(num_features, num_features,nn=nn_start, aggr='mean')
        self.conv_m = NNConv(num_features, num_features,nn=self.nn_middle, aggr='mean')
        self.conv2 = NNConv(num_features, 64,nn=nn_end, aggr='mean')
        self.multihead_attn = MultiheadAttention(num_features, 1)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_target)

    def forward(self, data):
       data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
       data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = self.nn_middle(F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr)))
       data.x, _ = self.multihead_attn(data.x, data.x, data.x)
       data.x = F.elu(self.conv_m(data.x, data.edge_index, data.edge_attr))
       data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))


       x = F.elu(self.fc1(data.x))
       x = F.dropout(x, training=self.training)
       return F.elu(self.fc2(x))


class HotEncoding(torch.nn.Module):
    
    def __init__(self,num_features = 10,num_target = 10,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        

        self.grap_conv = GraphConv(self.num_features, 10, aggr='mean')
        self.grap_conv_end = GraphConv(self.num_features, 10, aggr='mean')


        self.fc1 = torch.nn.Linear(10, 128)
        self.fc2 = torch.nn.Linear(128, self.num_target)
        
        self.v1 = torch.tensor( np.array([[1,0,0]]),dtype=torch.float).to(self.local_device)
    def forward(self, data):
 

        data.x = F.elu(self.grap_conv(data.x, data.edge_index))
        data.x = F.elu(self.grap_conv_end(data.x, data.edge_index))

        x = F.elu(self.fc1(data.x))
        return F.softmax(self.fc2(x))


class HotEncodingForLoopForward(torch.nn.Module):
    
    def __init__(self,num_features = 10,num_target = 10,local_device = 'cuda:0',param_dict={}):
        super().__init__()

        self.num_features = num_features
        self.num_target = num_target
        self.local_device = local_device
        

        self.grap_conv_begin = GraphConv(self.num_features, 10, aggr='mean')

        self.grap_conv_middle = GraphConv(self.num_features, 10, aggr='mean')
        self.grap_conv_end = GraphConv(self.num_features, 10, aggr='mean')


        self.fc1 = torch.nn.Linear(10, 128)
        self.fc2 = torch.nn.Linear(128, self.num_target)
        
        self.v1 = torch.tensor( np.array([[1,0,0]]),dtype=torch.float).to(self.local_device)
    def forward(self, data):
        data.x = F.elu(self.grap_conv_begin(data.x, data.edge_index))
        x_last = data.x.clone()
        for i in range(1):
            data.x = F.elu(self.grap_conv_middle(data.x, data.edge_index))
            #print(torch.dot(x_last.flatten(), data.x.flatten()).flatten()/(torch.norm(x_last.flatten())*torch.norm(data.x.flatten())))
            x_last = data.x.clone()
        data.x = F.elu(self.grap_conv_end(data.x, data.edge_index))

        x = F.elu(self.fc1(data.x))
        return F.softmax(self.fc2(x))
