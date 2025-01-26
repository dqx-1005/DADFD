#  DADFD: A Dual Adaptive Decoupled Fine-grained Distillation Framework for Graph Neural Networks

This is a PyTorch implementation of DADFD, and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-Phy, and ogbn-arxiv,ogbn-products)

* Various teacher GNN architectures (GCN, SAGE, GAT) and student GCNs, MLPs

* Training paradigm for teacher GNNs and student GCNs, MLPs


## Main Requirements

* numpy==1.21.6
* scipy==1.7.3
* torch==1.11.0
* dgl-cu111==0.6.1
* dcor==0.5.3
* matplotlib==3.5.3


## Description

* main.py  
  * Main script that handles the model training and evaluation process.

* teacher.py
  * Pre-train the teacher GNNs

* student.py
  * Train the student GCNs, MLPs with the pre-trained teacher GNNs
* models
  * MLP() -- student MLPs
  * GCN() -- GCN Classifier, working as teacher GNNs,student GNNs
  * GAT() -- GAT Classifier, working as teacher GNNs
  * GraphSAGE() -- GraphSAGE Classifier, working as teacher GNNs
  
* dataloader.py
  * load_data() -- Load Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-Phy, ogbn-arxiv, ogbn-products datasets

* utils.py  
  * mask_features() -- Randomly mask a portion of features
  * set_seed() -- Set radom seeds for reproducible results
  * graph_split() -- Split the data for the inductive setting


## Running the code

1. Install the required dependency packages
2. To reproduce the results in paper, please use the command in ./train.sh
