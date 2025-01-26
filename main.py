import dgl
import torch
from torch.utils.data import DataLoader
import argparse
import nni
import sys
import os
import json
import warnings
import time
from utils.metrics import set_seed
from pathlib import Path
from data.get_dataset import get_experiment_config
from data.utils import load_ogb_data, load_cpf_data, load_tensor_data
from dataloader import load_data
from utils.models import choose_teacher_model, choose_student_model, save_checkpoint
from teacher import train_teacher
from student import train_student
from models.GCN import GCN
from teacher import evalution
import csv
import numpy as np

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("finish")


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--batch_size', type=str, default=256, help="batch size used for training, validation and test")
    parser.add_argument('--max_epoch', type=str, default=500)
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--teacher_hidden', type=int, default=256, help='Teacher Model hidden')
    parser.add_argument('--teacher_layers', type=int, default=3, help='Teacher Model layer')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--student', type=str, default='GCN', help='student Model')
    parser.add_argument('--student_hidden', type=int, default=64, help='Student Model hidden')
    parser.add_argument('--student_layers', type=int, default=2, help='Student Model layers')
    parser.add_argument('--mode', type=str, default="stu", help='tea or stu')
    parser.add_argument("--K", type=int, default=5, help="k")
    args = parser.parse_args()

    param = args.__dict__
    param.update(nni.get_next_parameter())

    # read param from json
    if os.path.exists('./param/best_param.json'):
        param = json.loads(open('./param/best_param.json').read())[param['dataset']][param['teacher']][param['student']]

    param['device'] = device
    print(param)
    # load data
    if param['dataset'] in ['coauthor-cs', 'coauthor-phy', 'amazon-photo', 'cora', 'citeseer', 'pubmed', 'ogbn-arxiv',
                            'ogbn-product']:
        conf_data = []
    else:
        config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
        conf_data = get_experiment_config(config_data_path, dataset=args.dataset)
        conf_data['seed'] = args.seed
    print(conf_data)

    if param['dataset'] in ['ogbn-arxiv', 'ogbn-products']:
        param['norm_type'] = 'batch'
    else:
        param['norm_type'] = 'none'

    if param['dataset'] in ['amazon-photo', 'coauthor-phy', 'cora', 'citeseer', 'pubmed']:
        adj, adj_sp, G, features, labels, idx_train, idx_val, idx_test = load_cpf_data(dataset=param['dataset'],
                                                                                       dataset_path="./data/npz",
                                                                                       seed=param['seed'],
                                                                                       device=param['device'])
    elif param['dataset'] in ['ogbn-arxiv', 'ogbn-products']:
        # load data
        G, labels, idx_train, idx_val, idx_test = load_data(param['dataset'], dataset_path="./data",
                                                            split_idx=param['split_idx'], seed=param['seed'])

    G = G.to(param['device'])

    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    data = (idx_train, idx_val, idx_test)

    if args.mode == "tea":
        output_dir = os.path.join('./output', param['teacher'], param['dataset'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_save_path = os.path.join(output_dir, 'best_model.pt')

        best_acc = 0

        set_seed(param['seed'])
        # choose teacher model
        teacher_model = choose_teacher_model(param, G, labels)
        # choose teacher optimizer
        t_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=param['learning_rate'],
                                       weight_decay=param['weight_decay'])
        print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
        print("############ Teacher model#############")
        t_test_score, t_auc_score, t_f1_score = train_teacher(param, teacher_model, t_optimizer, data,
                                                              G,labels)
        # Check if the current model is the best
        if t_test_score > best_acc:
            best_acc = t_test_score
            # Save the best model
            torch.save(teacher_model.state_dict(), model_save_path)
            print(f"New best model saved with acc score: {t_test_score}")

    else:
        # trained teacher
        teacher_model = choose_teacher_model(param, G, labels)
        teacher_model_path = os.path.join('./output', param['teacher'], param['dataset'], 'best_model.pt')
        state_dict = torch.load(teacher_model_path)
        teacher_model.load_state_dict(state_dict)

    # train student
    if args.mode == "stu":
        best_student_model_score = 0
        best_student_model = None

        test_acc_list = []
        test_auc_list = []
        test_f1_list = []

        args.K = param['K']
        set_seed(param['seed'])
        student_model = choose_student_model(param, G, labels)
        print("############ Student model with Teacher #############")
        s_optimizer = torch.optim.Adam(list(student_model.parameters()),
                                       lr=param['learning_rate'],
                                       weight_decay=param['weight_decay'])
        test_acc, _, _, _ = train_student(param, student_model,
                                          teacher_model,
                                          s_optimizer, data, G
                                          , args, labels)

        if test_acc > best_student_model_score:
            best_student_model = student_model
            best_student_model_score = test_acc

        test_acc_list.append(torch.tensor(test_acc, device=param['device']))
    main()