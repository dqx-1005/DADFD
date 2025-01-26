import torch
import torch.nn as nn
import torch.nn.functional as F

class TransLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
    def forward(self, x):
        return F.linear(x, self.weight)

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation, dropout):
        super(LinearEncoder, self).__init__()
        self.linear_or_not = True
        self.activation = activation
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.trans = nn.ModuleList()
        for i in range(num_layers):
            self.trans.append(TransLayer(input_dim, output_dim))
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.linear_or_not = False
            # Input layer
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='relu')
            # Hidden layers
            for _ in range(num_layers - 2):
                layer = nn.Linear(hidden_dim, hidden_dim)
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                self.layers.append(layer)
            # Output layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            nn.init.kaiming_normal_(self.layers[-1].weight, nonlinearity='relu')

    def forward(self, x):
        mid_f = []

        if self.linear_or_not:
            x = self.layers[0](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            mid_f.append(x)
            return x,mid_f
        else:
            for layer in self.layers:
                x = layer(x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                mid_f.append(x)

            final_output = x
            return final_output, mid_f
    def loss_compute(self,output_t,output_s):
        loss_mse = 0
        for i in range(len(output_t)):
            loss_mse += torch.nn.functional.mse_loss(output_t[i]@self.trans[i].weight , output_s[i], reduction='mean')
        return (loss_mse/len(output_t)).item()