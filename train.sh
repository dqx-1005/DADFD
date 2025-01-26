#!/bin/bash
# cora
# train teacher
python main.py --dataset cora --teacher GCN --mode tea
python main.py --dataset cora --teacher GAT --mode tea
python main.py --dataset cora --teacher GraphSAGE --mode tea
# train student
# GCN
python main.py --dataset cora --teacher GCN --student GCN --mode stu
python main.py --dataset cora --teacher GAT --student GCN --mode stu
python main.py --dataset cora --teacher GraphSAGE --student GCN --mode stu
# MLP
python main.py --dataset cora --teacher GCN --student MLP --mode stu
python main.py --dataset cora --teacher GAT --student MLP --mode stu
python main.py --dataset cora --teacher GraphSAGE --student MLP --mode stu

# citeseer
# train teacher
python main.py --dataset citeseer --teacher GCN --mode tea
python main.py --dataset citeseer --teacher GAT --mode tea
python main.py --dataset citeseer --teacher GraphSAGE --mode tea
# train student
# GCN
python main.py --dataset citeseer --teacher GCN --student GCN --mode stu
python main.py --dataset citeseer --teacher GAT --student GCN --mode stu
python main.py --dataset citeseer --teacher GraphSAGE --student GCN --mode stu
# MLP
python main.py --dataset citeseer --teacher GCN --student MLP --mode stu
python main.py --dataset citeseer --teacher GAT --student MLP --mode stu
python main.py --dataset citeseer --teacher GraphSAGE --student MLP --mode stu

# pubmed
# train teacher
python main.py --dataset pubmed --teacher GCN --mode tea
python main.py --dataset pubmed --teacher GAT --mode tea
python main.py --dataset pubmed --teacher GraphSAGE --mode tea
# train student
# GCN
python main.py --dataset pubmed --teacher GCN --student GCN --mode stu
python main.py --dataset pubmed --teacher GAT --student GCN --mode stu
python main.py --dataset pubmed --teacher GraphSAGE --student GCN --mode stu
# MLP
python main.py --dataset pubmed --teacher GCN --student MLP --mode stu
python main.py --dataset pubmed --teacher GAT --student MLP --mode stu
python main.py --dataset pubmed --teacher GraphSAGE --student MLP --mode stu

# amazon-photo
# train teacher
python main.py --dataset amazon-photo --teacher GCN --mode tea
python main.py --dataset amazon-photo --teacher GAT --mode tea
python main.py --dataset amazon-photo --teacher GraphSAGE --mode tea
# train student
# GCN
python main.py --dataset amazon-photo --teacher GCN --student GCN --mode stu
python main.py --dataset amazon-photo --teacher GAT --student GCN --mode stu
python main.py --dataset amazon-photo --teacher GraphSAGE --student GCN --mode stu
# MLP
python main.py --dataset amazon-photo --teacher GCN --student MLP --mode stu
python main.py --dataset amazon-photo --teacher GAT --student MLP --mode stu
python main.py --dataset amazon-photo --teacher GraphSAGE --student MLP --mode stu

# coauthor-phy
# train teacher
python main.py --dataset coauthor-phy --teacher GCN --mode tea
python main.py --dataset coauthor-phy --teacher GAT --mode tea
python main.py --dataset coauthor-phy --teacher GraphSAGE --mode tea
# train student
# GCN
python main.py --dataset coauthor-phy --teacher GCN --student GCN --mode stu
python main.py --dataset coauthor-phy --teacher GAT --student GCN --mode stu
python main.py --dataset coauthor-phy --teacher GraphSAGE --student GCN --mode stu
# MLP
python main.py --dataset coauthor-phy --teacher GCN --student MLP --mode stu
python main.py --dataset coauthor-phy --teacher GAT --student MLP --mode stu
python main.py --dataset coauthor-phy --teacher GraphSAGE --student MLP --mode stu

# ogbn-arxiv
# train teacher
python main.py --dataset ogbn-arxiv --teacher GCN --mode tea
# train student
# MLP
python main.py --dataset ogbn-arxiv --teacher GCN --student MLP --mode stu

# ogbn-products
# train teacher
python main.py --dataset ogbn-products --teacher GCN --mode tea
# train student
# MLP
python main.py --dataset ogbn-products --teacher GCN --student MLP --mode stu
