import os
import sys
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
from data.io_dataset import load_dataset
from data.preprocess import eliminate_self_loops,binarize_labels,to_binary_bag_of_words

def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry ==1.0 for _,_,single_entry in zip(features_coo.row,features_coo.col,features_coo.data))

def get_dataset(name,data_path,standardize,train_examples_per_class=None,val_examples_per_class=None):
    dataset_graph = load_dataset(data_path)
    print("standardize：",standardize)
    if standardize:
        dataset_graph = dataset_graph.standardize(name)
        #print(dataset_graph.adj_matrix)
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)
    '''
    if train_examples_per_class is not None and val_examples_per_class is not None:
        if name == 'cora_full':
            # cora_full has some classes that have very few instances. We have to remove these in order for
            # split generation not to fail
            # wait for writing
    '''
    graph_adj,node_features,labels = dataset_graph.unpack()
    labels = binarize_labels(labels)

    print('graph_adj',graph_adj.shape)

    if not is_binary_bag_of_words(node_features):
        node_features = to_binary_bag_of_words(node_features)

    assert (graph_adj!=graph_adj.T).nnz == 0

    assert is_binary_bag_of_words(node_features),f'Non-binary node_features entry!'

    return graph_adj,node_features,labels




def sample_per_class(random_state,labels,num_examples_per_class,forbidden_indices=None):
    num_samples,num_classes = labels.shape
    sample_indices_per_class = {
        index:[] for index in range(num_classes)
    }

    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index,class_index]>0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index],num_examples_per_class,replace=False)
         for class_index in range(len(sample_indices_per_class))]
    )

def get_dataset_and_split_planetoid(dataset,data_path):
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    # if _log is not None:
    #     _log.info('Loading dataset %s.' % dataset)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(data_path, "ind.{}.test.index".format(dataset))
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # cast!!!
    # adj = adj.astype(np.float32)
    # features = features.tocsr()
    # features = features.astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test


def get_train_val_test_split(random_state,labels,
                             train_examples_per_class=None,
                             val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None,val_size=None,test_size=None,
                             ):
    num_samples,num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    #print(train_examples_per_class,val_examples_per_class,test_examples_per_class,train_size,val_size,test_size)
    print(f"Train examples per class: {train_examples_per_class}")
    print(f"Validation examples per class: {val_examples_per_class}")
    print(f"Test examples per class: {test_examples_per_class}")
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state,labels,train_examples_per_class)
    else:
        train_indices = random_state.choice(remaining_indices,train_size,replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state,labels,val_examples_per_class,forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices,train_indices)
        val_indices = random_state.choice(remaining_indices,val_size,replace = False)

    forbiddent_indices = np.concatenate((train_indices,val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state,labels,test_examples_per_class,forbidden_indices=forbiddent_indices)
    elif test_size is not None:
        remaining_indices =  np.setdiff1d(remaining_indices,forbiddent_indices)
        test_indices = random_state.choice(remaining_indices,test_size,replace = False)
    else:
        test_indices = np.setdiff1d(remaining_indices,forbiddent_indices)

    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices))   == len(val_indices)
    assert len(set(test_indices))  == len(test_indices)
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))

    if test_size is None and test_examples_per_class is None:
        assert len(np.concatenate((train_indices,val_indices,test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices,:]
        train_sum = np.sum(train_labels,axis=0)
        assert np.unique(train_sum).size ==1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices,:]
        val_sum  = np.sum(val_labels,axis=0)
        assert np.unique(val_sum).size ==1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices,:]
        test_sum = np.sum(test_labels,axis=0)
        assert np.unique(test_sum).size ==1

    return train_indices,val_indices,test_indices






