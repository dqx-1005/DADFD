import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2,output_dim, num_layers, activation, dropout):
        super(LinearDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim_1))
        nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='relu')

        # Second layer
        self.layers.append(nn.Linear(hidden_dim_1, hidden_dim_2))
        nn.init.kaiming_normal_(self.layers[-1].weight, nonlinearity='relu')

        # Output layer
        self.layers.append(nn.Linear(hidden_dim_2, output_dim))
        nn.init.kaiming_normal_(self.layers[-1].weight,nonlinearity='linear')

    def forward(self, x):
        intermediate_outputs = []
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            intermediate_outputs.append(x)

        final_output = self.layers[-1](x)
        return final_output, intermediate_outputs