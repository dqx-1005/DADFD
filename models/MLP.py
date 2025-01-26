import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers,input_dim,hidden_dim,output_dim,dropout,norm_type="none"):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer).
            If num_layers=1-1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
        super(MLP,self).__init__()

        self.linear_or_not = True
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=1-dropout)
        self.norms = nn.ModuleList()
        self.norm_type = norm_type

        if num_layers<1:
            raise ValueError("number of layers should be positive!")
        elif num_layers==1:
            self.layers = nn.Linear(input_dim,output_dim)
        else:
            self.linear_or_not = False
            self.layers = torch.nn.ModuleList()
            self.layers.append(nn.Linear(input_dim,hidden_dim))
            if self.norm_type=='batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type=='layer':
                self.norms.append(nn.LayerNorm(hidden_dim))

            for layer in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                if self.norm_type=='batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type=='layer':
                    self.norms.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.Linear(hidden_dim,output_dim))

    def forward(self, feats):
        mid_f = []
        penultimate_feats = None
        if self.linear_or_not:
            return F.relu(self.layers(self.dropout(feats))),mid_f,penultimate_feats
        else:
            h = feats
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i!=self.num_layers-1:
                    if self.norm_type!="none":
                        h = self.norms[i](h)
                    h = F.relu(h)
                    h = self.dropout(h)
                if i == self.num_layers - 2:
                    penultimate_feats = h
                mid_f.append(h)
            return h,mid_f,penultimate_feats