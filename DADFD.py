import torch
from disc import top_k_mask,split_into_k_groups,pca_with_mse,coral
import math
import numpy as np
import torch.nn.functional as F

def DADFD(param, data, G, t_penultimate_f, s_penultimate_f, output_t, output_s, encoder, decoder,tsgap,args,labels,max_k):
    train, val, test = data

    if param['dataset'] in ['cora', 'citeseer', 'pubmed', 'coauthor-cs', 'coauthor-phy', 'amazon-photo']:
        train = train.cpu().numpy()
    device = param['device']

    multiple_ts = torch.cat((t_penultimate_f, s_penultimate_f), dim=1)
    encoder.train()
    output_encoder, mid_encoder = encoder(multiple_ts)

    separator = 0.5
    tsgap_tensor = torch.tensor(tsgap, device=device)
    tsgap_tensor = tsgap_tensor.squeeze()
    k_values = (tsgap_tensor // separator).long()
    k_values = torch.minimum(k_values, torch.tensor(max_k - 1, device=device))

    s_concat_tensors = []
    for k in range(max_k):
        indices = torch.where(k_values == k)[0]
        if len(indices) > 0:
            s_concat_tensors.append(s_penultimate_f[indices])
        else:
            s_concat_tensors.append(torch.empty((0, param['student_hidden']), device=device))

    rts_tensors_list = [torch.empty((0, param['student_hidden'] * (k + 1)), device=device) for k in range(max_k)]
    k_values_tensor = torch.tensor(k_values, device=device)

    for k in range(max_k):
        node_indices = (k_values_tensor == k).nonzero(as_tuple=True)[0]
        if len(node_indices) > 0:
            concat_feats = torch.cat([mid_encoder[layer][node_indices, :] for layer in range(k + 1)], dim=-1)
            rts_tensors_list[k] = concat_feats

    mse_scores = []
    for k in range(max_k):
        mse_scores.append(pca_with_mse(s_concat_tensors[k], rts_tensors_list[k], param))
    mse_tensor = torch.stack(mse_scores)

    dec_score = torch.mean(mse_tensor)
    dec_score = dec_score.item()
    dec_score = 1 / (math.log(1 + dec_score) + 1)

    # calculate diversity
    diversity_sum = 0
    for i in range(len(mid_encoder)):
        for j in range(0, i, 1):
            if i != j:
                if rts_tensors_list[i].shape[0] <= 1 or rts_tensors_list[j].shape[0] <= 1:
                    continue
                d_score = coral(rts_tensors_list[i], rts_tensors_list[j], param)
                if np.isnan(d_score.item()):
                    print("error", rts_tensors_list[i].shape, " , ", rts_tensors_list[j].shape)
                diversity_sum += d_score.item()
    if max_k == 1:
        diversity_score = diversity_sum
    else:
        diversity_score = diversity_sum / (max_k * (max_k - 1) / 2)
    diversity_score = 1.0 / (1 + diversity_score ** 0.5)
    if np.isnan(diversity_score):
        print("error: ", diversity_sum)

    # concat
    tensor_list = [torch.tensor(arr) for arr in mid_encoder]
    concatenated_tensor = torch.cat(tensor_list, dim=1)

    # decoder
    decoder.train()
    decoder_output, decoder_mid = decoder(concatenated_tensor.to(param["device"]))

    # ReKD
    if param['student'] == 'GCN':
        T = 1.0
        output_s = output_s[train]
        output_t = output_t[train]
        k_values = k_values[train]
        decoder_output = decoder_output[train]
    elif param['student'] == 'MLP':
        T = 0.7

    s_mask = top_k_mask(output_s,k_values,max_k)
    s_topk_logit = output_s / T - 1000 * s_mask
    s_topk_logit = F.log_softmax(s_topk_logit, dim=1)
    t_topk_logit = output_t / T - 1000 * s_mask
    t_topk_logit = F.softmax(t_topk_logit, dim=1)
    rt_topk_logit = decoder_output / T - 1000 * s_mask
    rt_topk_logit = F.softmax(rt_topk_logit, dim=1)

    L_TISD = F.kl_div(s_topk_logit, t_topk_logit, size_average=False) * (T * T) / \
             output_s.size()[0]

    not_s_mask = 1 - s_mask
    s_not_topk_logit = output_s / T - 1000 * not_s_mask
    t_not_topk_logit = output_t / T - 1000 * not_s_mask
    rt_not_topk_logit = decoder_output / T - 1000 * not_s_mask
    s_logsoftmax = F.log_softmax(s_not_topk_logit, dim=1)
    t_softmax = F.softmax(t_not_topk_logit, dim=1)
    rt_softmax = F.softmax(rt_not_topk_logit, dim=1)

    L_NTID = F.kl_div(s_logsoftmax, t_softmax, size_average=False) * (T * T) / output_s.size()[0]

    if param['dataset'] in ['cora', 'citeseer']:
        rekd_alpha = 15
        rekd_beta = 3
    elif param['dataset'] in ['pubmed']:
        rekd_alpha = 18
        rekd_beta = 3
    elif param['dataset'] in ['amazon-photo']:
        rekd_alpha = 24
        rekd_beta = 1
    elif param['dataset'] in ['ogbn-arxiv']:
        rekd_alpha = 1
        rekd_beta = 1
    elif param['dataset'] in ['ogbn-products']:
        rekd_alpha = 1
        rekd_beta = 1
    else:
        rekd_alpha = 15
        rekd_beta = 3

    loss_kd = rekd_alpha*L_TISD + rekd_beta*L_NTID

    kl_loss = torch.nn.KLDivLoss(reduction='none')
    loss_CE = torch.nn.CrossEntropyLoss(reduction='none')
    output_t = output_t.to(param['device'])
    labels = labels.to(param['device'])

    if param['student'] == 'GCN':
        t_s_kl = kl_loss(s_topk_logit, t_topk_logit)
        rt_s_kl = kl_loss(s_topk_logit, rt_topk_logit)
        t_ce_loss = loss_CE(output_t, labels[train])
        r_ts_ce_loss = loss_CE(decoder_output, labels[train])
    else:
        t_s_kl = kl_loss(s_topk_logit[train], t_topk_logit[train])
        rt_s_kl = kl_loss(s_topk_logit[train], rt_topk_logit[train])
        t_ce_loss = loss_CE(output_t[train], labels[train])
        r_ts_ce_loss = loss_CE(decoder_output[train], labels[train])

    kl_diff = (rt_s_kl - t_s_kl).mean()
    kl_hinge_loss = torch.clamp(kl_diff, min=0.2)
    kl_hinge_score = kl_hinge_loss.mean()

    ce_diff = (r_ts_ce_loss - t_ce_loss).mean()
    ce_hinge_loss = torch.clamp(ce_diff, min=0.2)
    ce_loss_score = ce_hinge_loss.mean()

    rec_score = (ce_loss_score + kl_hinge_score) / 2

    return dec_score, diversity_score, rec_score, ce_loss_score, kl_hinge_score,loss_kd
