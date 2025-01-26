import torch
import torch.nn.functional as F
import dgl
import numpy as np
import time
from utils.metrics import accuracy,F1,multi_label_auc,ogbn_acc
import copy

def train_teacher(param,teacher_model,optimizer,data,G,labels):

    train,valid,test = data
    time_mark = []
    t0 = time.time()

    best = 0
    early_stop = 0
    loss_CE = torch.nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(param['max_epoch']):
        teacher_model.train()
        feat = G.ndata['feat'].to(param['device'])
        labels = labels.to(param['device'])
        teacher_model.g = G

        for layer in teacher_model.layers:
            layer.g = G

        if param['teacher'] in ['GCN','GraphSAGE']:
            output_t,mid_t, penultimate_t = teacher_model(feat)
        elif param['teacher'] in ['GAT']:
            output_t,_,mid_t, penultimate_t = teacher_model(feat)
        else:
            raise ValueError(f'Undefined Model')

        loss = loss_CE(output_t.log_softmax(dim=1)[train], labels[train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_data = np.array(loss.item())

        time_mark.append(time.time() - t0)
        train_acc,_,_ = evalution(param,teacher_model,train,G,labels)
        val_acc,_,_ = evalution(param, teacher_model, valid, G,labels)

        print(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) +
              'Epoch %d | Loss: %.4f | train_acc: %.4f | val_acc: %.4f | Time(s) %.4f'
              % (epoch, loss_data.item(), train_acc.item(), val_acc.item(), time_mark[-1]))

        if val_acc > best:
            best = val_acc
            state = dict([('model', copy.deepcopy(teacher_model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            early_stop = 0
        else:
            early_stop += 1
        if epoch == param['max_epoch'] or early_stop == 50:
            print("Stop!!!")
            break

    teacher_model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])

    test_score, auc_score, f1_score = evalution(param, teacher_model, test, G,labels)

    print(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) + " The teacher's test set results: acc_test= %.4f" % (test_score))
    print(str(time.strftime("[%Y-%m-%d %H:%M:%S]",
                            time.localtime())) + " The teacher's test set results: auc_test= %.4f" % (auc_score))
    print(str(time.strftime("[%Y-%m-%d %H:%M:%S]",
                            time.localtime())) + " The teacher's test set results: f1_test= %.4f" % (f1_score))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time_mark[-1]))

    return test_score,auc_score,f1_score

def evalution(param,teacher_model,data,G,labels):
    teacher_model.eval()
    with torch.no_grad():
        feat = G.ndata['feat'].to(param['device'])
        labels = labels.to(param['device'])

        teacher_model.g = G
        if param['teacher'] in ['GCN','GAT','GraphSAGE']:
            for layer in teacher_model.layers:
                layer.g = G
            output_t, t_mid,_ = teacher_model(feat)
        if param['teacher'] in ['GCN','GraphSAGE']:
            output_t,mid_t, penultimate_t = teacher_model(feat)
        elif param['teacher'] in ['GAT']:
            output_t, _,mid_t, penultimate_t = teacher_model(feat)
        else:
            raise ValueError(f'Undefined Model')

        logp = F.log_softmax(output_t, dim=1)
        if param['dataset'] in ["ogbn-arxiv", "ogbn-products"]:
            acc = ogbn_acc(param['dataset'],logp[data],labels[data])
        else:
            acc = accuracy(logp[data],labels[data])

        auc = multi_label_auc(labels[data].detach().cpu().numpy(),logp[data].detach().cpu().numpy())
        f1 = F1(logp[data],labels[data])

    return acc,auc,f1



