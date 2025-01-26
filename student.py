import math
import os
import time
import torch
import copy
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models import GCN
from utils.metrics import *
from utils.models import choose_encoder, choose_decoder
import csv
import logging
from disc import *
from collections import defaultdict
from DADFD import DADFD


def train_student(param, student_model, teacher_model, optimizer, data, G, args,labels):
    set_seed(param['seed'])
    train, val, test = data
    if param['dataset'] in ['coauthor-cs', 'amazon-photo', 'cora', 'coauthor-phy']:
        train = train.cpu().numpy()
        val = val.cpu().numpy()
        test = test.cpu().numpy()

    device = param['device']
    time_mark = []
    t0 = time.time()
    max_k = args.K

    # init encoder
    encoder = choose_encoder(param, param['teacher_hidden'] + param['student_hidden'], max_k)

    if param['dataset'] in ['ogbn-arxiv', 'ogbn-products']:
        label = G.ndata['label'].to(device)
    else:
        label = labels

    decoder = choose_decoder(param, param['student_hidden'] * max_k, label)

    all_params = []
    all_params.extend(list(encoder.parameters()))
    all_params.extend(list(decoder.parameters()))

    encoder_decoder_optimizer = torch.optim.Adam(all_params, lr=0.01, weight_decay=5e-4)

    best_test_acc = 0
    best = 0
    early_stop = 0

    loss_CE = torch.nn.CrossEntropyLoss(reduction='mean')

    teacher_model.eval()
    feat = G.ndata['feat'].to(device)

    label = label.to(device)
    G = G.to(device)

    teacher_model.g = G
    for layer in teacher_model.layers:
        layer.g = G

    with torch.no_grad():
        if param['teacher'] in ['GAT']:
            output_t, _, t_middle_f, t_penultimate_f = teacher_model(feat)
        else:
            output_t, t_middle_f, t_penultimate_f = teacher_model(feat)

    for epoch in range(param['max_epoch']):
        student_model.train()
        if param['student'] == 'GCN':
            student_model.g = G
            for layer in student_model.layers:
                layer.g = G
            output_s, s_middle_f, s_penultimate_f = student_model(feat)

        if param['student'] == 'MLP':
            output_s, s_middle_f, s_penultimate_f = student_model(feat)

        # first epoch
        if epoch == 0:
            tsgap = torch.full((G.ndata['feat'].shape[0], 1), 10)
        else:
            label = label.to(device)
            output_s = output_s.to(device)
            output_t = output_t.to(device)

            correct_output_s = output_s.gather(1, label.unsqueeze(1))
            correct_output_t = output_t.gather(1, label.unsqueeze(1))
            # the gap between teacher and student
            tsgap = F.mse_loss(correct_output_s, correct_output_t, reduction='none')

        dec_score, diversity_score, rec_score, ce_loss_score, kl_hinge_score, loss_kd = \
            DADFD(param, data, G, t_middle_f[-2], s_middle_f[-2], output_t,output_s, encoder, decoder, tsgap, args, labels, max_k)

        output_s = output_s.to(device)

        loss_ce = loss_CE(output_s[train], label[train])

        # val
        loss_ce_val = loss_CE(output_t[val], label[val])
        loss_kd_val = F.mse_loss(output_s[val], output_t[val])


        loss = param['alpha'] * loss_ce + (1 - param['alpha']) * loss_kd + param['beta'] * dec_score + param[
                'gamma'] * diversity_score + param['delta'] * rec_score

        # KD
        loss_val = param['alpha'] * loss_ce_val + (1 - param['alpha']) * loss_kd_val

        optimizer.zero_grad()
        encoder_decoder_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        encoder_decoder_optimizer.step()

        loss_data = np.array(loss.item())
        time_mark.append(time.time() - t0)

        train_score, _, _ = evalution(param, student_model, train, G, labels)
        val_score, _, _ = evalution(param, student_model, val, G, labels)

        if not isinstance(dec_score, float):
            dec_score = float(dec_score)
        if not isinstance(diversity_score, float):
            diversity_score = float(diversity_score)
        if not isinstance(rec_score, float):
            rec_score = float(rec_score)
        print(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) +
                  'Epoch %d | loss_total: %.4f | loss_val: %.4f | Loss_ce: %.4f | Loss_kd: %.4f | Time(s) %.4f' %
                  (epoch, loss_data.item(), loss_val, loss_ce.item(), loss_kd.item(),time_mark[-1]))
        test_acc, test_auc_best, test_f1_best = evalution(param, student_model, test, G, labels)
        if test_acc > best_test_acc: best_test_acc = test_acc
        print('dec_score: %.4f |diversity_score: %.4f |rec_score: %.4f |train_score: %.4f |val_score: %.4f | test_score: %.4f' % (
                dec_score, diversity_score, rec_score,train_score.item(), val_score.item(), test_acc))

        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        print('total time: ', time.time() - t0)

        if val_score > best:
            best = val_score
            state = dict([('student_model', copy.deepcopy(student_model.state_dict())),
                          ('optim', copy.deepcopy(encoder_decoder_optimizer.state_dict()))])
            # print("reset early stop")
            early_stop = 0
        else:
            early_stop += 1

        if epoch == param['max_epoch'] or early_stop == 50:
            print("Stop!!!")
            break

    student_model.load_state_dict(state['student_model'])
    encoder_decoder_optimizer.load_state_dict(state['optim'])

    test_acc, test_auc_best, test_f1_best = evalution(param, student_model, test, G, labels)
    print("Test set results: test_acc= {:.4f} | test_auc= {:.4f} | test_f1= {:.4f}".format(best_test_acc.item(),
                                                                                           test_auc_best.item(),
                                                                                           test_f1_best.item()))

    return best_test_acc, test_auc_best, test_f1_best, epoch


def evalution(param, student_model, data, G, labels):
    student_model.eval()
    with torch.no_grad():
        feat = G.ndata['feat'].to(param['device'])
        label = labels
        student_model.g = G
        for layer in student_model.layers:
            layer.g = G

        logits, middle_f, penultimate_f = student_model(feat)

        logp = F.log_softmax(logits, dim=1)
        if param['dataset'] in ["ogbn-arxiv", "ogbn-products"]:
            test_acc = ogbn_acc(param['dataset'], logp[data], label[data])
        else:
            test_acc = accuracy(logp[data], label[data])

        test_auc = multi_label_auc(label[data].detach().cpu().numpy(), logits[data].detach().cpu().numpy())
        test_f1 = F1(logp[data], label[data])

        return test_acc, test_auc, test_f1
