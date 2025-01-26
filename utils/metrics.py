import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score,roc_auc_score
import dcor
from ogb.nodeproppred import Evaluator

def pad_tensor(tensor, target_length):
    """
    fill the tensor with padded values
    """
    shape = tensor.shape
    padding = target_length - shape[1]
    if padding > 0:
        pad = (0, padding)
        padded_tensor = F.pad(tensor, pad, mode='constant', value=0)
        return padded_tensor
    else:
        return tensor

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x):

    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

def accuracy(output, labels,details=False,hop_idx=None):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()

    if details:
        hop_sum = np.bincount(hop_idx,minlength=7)
        true_idx = np.array((correct>0).nonzero().squeeze(dim=1).cpu())
        true_hop = np.bincount(hop_idx[true_idx],minlength=7) / hop_sum
        return result/len(labels),true_hop
    acc = result / len(labels)
    return acc

def F1(output, labels):
    y_true = labels.detach().cpu().numpy()
    y_pred = output.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true,y_pred, average='macro')
    return f1

def multi_label_auc(true,scores):
    y_scores=softmax(scores)
    if len(np.unique(y_scores))!=y_scores.shape[1]:
        auc = roc_auc_score(true,y_scores,multi_class='ovo', labels=np.linspace(0, y_scores.shape[1]-1, y_scores.shape[1]))
    else:
        auc = roc_auc_score(true,y_scores,multi_class='ovo')
    return auc

def ogbn_acc(dataset,out,labels):
    ogb_evaluator = Evaluator(dataset)
    pred = out.argmax(1,keepdim=True)
    input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
    acc = ogb_evaluator.eval(input_dict)["acc"]
    return acc