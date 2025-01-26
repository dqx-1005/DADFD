import dgl
import scipy.sparse as sp
import torch
import numpy as np
import os
from pathlib import Path
from data.io_dataset import load_npz_to_sparse_graph
from data.preprocess import binarize_labels
from data.make_dataset import get_train_val_test_split
from data.get_dataset import load_dataset_and_split

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def check_isolated_nodes(adj):
    """Check for isolated nodes in the adjacency matrix."""
    rowsum = np.array(adj.sum(1)).flatten()
    isolated_nodes = np.where(rowsum == 0)[0]
    return isolated_nodes

def normalize_adj(adj):
    isolated_nodes = check_isolated_nodes(adj)
    print(f"Isolated nodes before adding self-loops: {isolated_nodes}")
    adj = normalize(adj+sp.eye(adj.shape[0]))
    adj_sp =(sp.csr_matrix(adj)).tocoo()
    return adj,adj_sp

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords,values,shape

    if isinstance(sparse_mx,list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    return normalize_adj(adj)

def preprocess_features(features):
    features = normalize(features)
    return features,sparse_to_tuple(features)

def load_ogb_data(dataset,device):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(name=dataset,root='data')
    splitted_idx = data.get_idx_split()
    idx_train = splitted_idx['train']
    idx_val = splitted_idx['valid']
    idx_test = splitted_idx['test']
    g, labels = data[0]
    features = g.ndata['feat']
    labels = labels.squeeze()

    # Turn the graph to undirected
    if dataset == "ogbn-arxiv":
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()

    return g, features, labels, idx_train, idx_val, idx_test

def load_cpf_data(dataset,dataset_path,seed,device,labelrate_train=20,labelrate_val=30):
    data_path = Path.cwd().joinpath(dataset_path,f'{dataset}.npz')
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f'{dataset_path} does not exist')

    data = data.standardize(dataset)
    adj,features,labels = data.unpack()

    labels = binarize_labels(labels)

    random_state = np.random.RandomState(seed)
    idx_train,idx_val,idx_test = get_train_val_test_split(
        random_state,labels,labelrate_train,labelrate_val
    )

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels.argmax(axis=1))
    adj = normalize_adj(adj)

    if dataset in['coauthor-cs','amazon-photo','cora','citeseer','pubmed','coauthor-phy']:
        adj_sp = adj[1]
    else:
        adj_sp = adj.tocoo()

    g = dgl.graph((adj_sp.row,adj_sp.col))
    g.ndata['feat'] = features

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    g = g.to(device)

    features = features.to(device)
    labels = labels.to(device)

    return adj,adj_sp,g,features,labels,idx_train,idx_val,idx_test


def load_tensor_data(model_name,dataset,conf_data,device):
    adj,features,labels_one_hot,idx_train,idx_val,idx_test = load_dataset_and_split(conf_data,dataset)
    adj,adj_sp = normalize_adj(adj)
    adj = sp.csr_matrix(adj)

    features,_ = preprocess_features(features)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = labels_one_hot.argmax(axis=1)

    labels = torch.LongTensor(labels)
    labels_one_hot = torch.FloatTensor(labels_one_hot)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    labels_one_hot = labels_one_hot.to(device)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj,adj_sp,features,labels,labels_one_hot,idx_train,idx_val,idx_test