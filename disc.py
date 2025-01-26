import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.cm import get_cmap
from scipy.interpolate import interp1d

import matplotlib
from scipy.interpolate import interp1d

# use fixed K
def split_into_k_groups(ts_gap, K):
    tsgap_tensor = ts_gap.squeeze()
    sorted_indices = torch.argsort(tsgap_tensor)
    num_samples = len(tsgap_tensor)
    base_size = num_samples // K
    remainder = num_samples % K

    category_labels = torch.zeros(num_samples, dtype=torch.long)
    labels = torch.zeros(num_samples, dtype=torch.long)
    index = 0

    for i in range(K):
        count = base_size + (1 if i < remainder else 0)
        labels[index:index + count] = i
        index += count

    category_labels[sorted_indices] = labels

    return category_labels

def pca(Y2, Y1=None, param=None, k=0):
    Y2_mean = Y2.mean(dim=0)
    Y2_centered = Y2 - Y2_mean

    cov_matrix = torch.mm(Y2_centered.T, Y2_centered) / (Y2_centered.size(0) - 1)
    cov_matrix += torch.eye(cov_matrix.size(0), device=param['device']) * 1e-4

    if Y2.size(0) == 1:
        cov_matrix = torch.zeros(Y2.size(1), Y2.size(1), device=param['device'])

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)

    if k == 0:
        if param is not None and 'student_hidden' in param:
            k = param['student_hidden']
        else:
            k = eigenvalues.size(0)

    k = min(k, eigenvectors.size(1))
    top_k_eigenvectors = eigenvectors[:, sorted_indices[:k]]
    Y2_reduced = torch.mm(Y2_centered, top_k_eigenvectors)

    if Y1 is not None:
        mse = torch.nn.functional.mse_loss(Y1, Y2_reduced, reduction="mean")
        return Y2_reduced, mse
    else:
        return Y2_reduced


'''
coral loss
'''
def coral(source, target,param):
    k = min(source.size(1),target.size(1))
    if source.size(1)!=k:
        source_pca = pca(source, None,param,k)
    else:
        source_pca = source
    if target.size(1)!=k:
        target_pca = pca(target,None,param, k)
    else:
        target_pca = target

    d = source.size(1)
    source_c = compute_covariance(source_pca)
    target_c = compute_covariance(target_pca)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)

    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

# ReKD
# 0为top K,1为非top K
def top_k_mask(s_output, k_value,top_k):
    batch_size, num_columns = s_output.shape
    k_value = torch.clamp(k_value, max=top_k)
    sorted_indices = torch.argsort(s_output, dim=1, descending=True)
    col_indices = torch.arange(num_columns, device=s_output.device).unsqueeze(0).expand(batch_size, -1)
    mask = col_indices < k_value.unsqueeze(1)
    s_mask = torch.ones_like(s_output, dtype=torch.int)
    s_mask.scatter_(1, sorted_indices, (1 - mask.int()))

    return s_mask

def interpolate_tensor(tensor, target_cols):
    rows, cols = tensor.shape
    original_x = np.linspace(0, cols - 1, cols)
    target_x = np.linspace(0, cols - 1, target_cols)

    interpolated_data = np.array(
        [interp1d(original_x, tensor[i].cpu().detach().numpy(), kind='linear')(target_x) for i in range(rows)]
    )
    return torch.tensor(interpolated_data, device=tensor.device)

# pca with mse
def pca_with_mse(Y1, Y2, param, k=0):
    Y2_mean = Y2.mean(dim=0)
    Y2_centered = Y2 - Y2_mean

    cov_matrix = torch.mm(Y2_centered.T, Y2_centered) / (Y2_centered.size(0) - 1)

    cov_matrix += torch.eye(cov_matrix.size(0), device=param['device']) * 1e-4

    # 一行数据无法进行奇异值分解
    if Y2.shape[0]<=1:
        return torch.tensor(0).to(param['device'])

    U, S, V = torch.svd(cov_matrix)

    if k == 0:
        k = param['student_hidden']

    top_k_eigenvectors = U[:, :k]

    Y2_reduced = torch.mm(Y2_centered, top_k_eigenvectors)

    if Y1.shape != Y2_reduced.shape:
        raise ValueError(f"Shape mismatch: Y1 shape {Y1.shape} and Y2_reduced shape {Y2_reduced.shape}")

    for i in range(len(Y2)):
        mse = torch.nn.functional.mse_loss(Y1[i], Y2_reduced[i], reduction="mean")
        if(mse.numel()==0):continue

    return mse