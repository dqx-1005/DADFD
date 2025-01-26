import os
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.LinearEncoder import LinearEncoder
from models.LinearDecoder import LinearDecoder
from models.MLP import MLP
import torch
import torch.nn.functional as F

def choose_teacher_model(param,G,labels):
    if param['teacher'] == 'GCN':
        model = GCN(
            g = G,
            in_feats = G.ndata['feat'].shape[1],
            n_hidden= param['teacher_hidden'],
            n_classes=labels.max().item()+1,
            n_layers=param['teacher_layers'],
            activation=F.relu,
            dropout=param['dropout_t']).to(param['device'])

    elif param['teacher'] in ['GAT', 'SGAT']:
        if param['teacher'] == 'GAT':
            num_heads = 8
        else:
            num_heads = 1
        num_layers = 1
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=G.ndata['feat'].shape[1],
                    num_hidden=int(param['teacher_hidden'])//num_heads,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,
                    residual=False).to(param['device'])
    elif param['teacher'] == 'GraphSAGE':
        model = GraphSAGE(g=G,in_feats=G.ndata['feat'].shape[1],
                          n_hidden=param['teacher_hidden'],
                          n_classes=labels.max().item() + 1,
                          n_layers=param['teacher_layers'],
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type='gcn').to(param['device'])
    return model

def choose_student_model(param,G,labels):
    if param['student'] == 'GCN':
        model = GCN(
            g = G,
            in_feats=G.ndata['feat'].shape[1],
            n_hidden=param['student_hidden'],
            n_classes=labels.max().item() + 1 ,
            n_layers=param['student_layers'],
            activation=F.relu,
            dropout=param['dropout_s']).to(param['device'])
    elif param['student'] == 'MLP':
        model = MLP(
            num_layers=param['student_layers'],
            input_dim=G.ndata['feat'].shape[1],
            hidden_dim=param['student_hidden'],
            output_dim=labels.max().item() + 1,
            dropout=param['dropout_s'],
            norm_type=param['norm_type']).to(param['device'])

    return model

def choose_encoder(param,input_dim,max_k):
    model = LinearEncoder(
        input_dim=input_dim,
        hidden_dim=param['student_hidden'],
        output_dim=param['student_hidden'],
        num_layers=max_k,
        activation=F.relu,
        dropout=0.2).to(param['device'])
    return model

def choose_decoder(param,input_dim,labels):
    model = LinearDecoder(
        input_dim=input_dim,
        hidden_dim_1=param['teacher_hidden'],
        hidden_dim_2=param['student_hidden'],
        output_dim=labels.max().item() + 1 ,
        num_layers= 3,
        activation=F.relu,
        dropout=0.2).to(param['device'])
    return model

# save model
def save_checkpoint(model,path):
    dirname = os.path.isdirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(),path)
    print(f"save model to {path}")