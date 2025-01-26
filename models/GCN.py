import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()

        self.g = g
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        if n_layers == 1:
            self.layers.append(GraphConv(in_feats, n_classes, activation=activation))
        else:
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
            for i in range(n_layers - 2):
                self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
            self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout =dropout

    def forward(self, features):
        h = features
        middle_feats = []
        penultimate_feats = None

        for i, layer in enumerate(self.layers):
            if i != 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = layer(self.g, h)
            middle_feats.append(h)
            if i == len(self.layers) - 2:
                penultimate_feats = (h)
        return h, middle_feats, penultimate_feats