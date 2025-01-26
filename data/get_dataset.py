import yaml
import argparse
import numpy as np

from data.make_dataset import get_dataset,get_train_val_test_split,get_dataset_and_split_planetoid
def get_experiment_config(config_path,dataset):
    with open(config_path,'r') as conf:
        conf_data = yaml.load(conf,Loader = yaml.FullLoader)
    specific_config = dict(**conf_data[dataset])
    specific_config['dataset_source'] = conf_data['dataset_source']
    return specific_config

def generate_data_path(dataset,dataset_source):

    if dataset_source =='planetoid':
        return 'data/planetoid'
    elif dataset_source =='npz':
        return 'data/npz/' + dataset + '.npz'
    else:
        print(dataset_source)
        raise ValueError(f'The {dataset} must be set to "planetoid" or "npz"')

def load_dataset_and_split(data_conf,dataset):
    _config = data_conf
    # _config = {
    #     'dataset_source': 'npz',
    #     'seed': 0,
    #     'train_config': {
    #         'split': {
    #             'train_examples_per_class': 20,  # 20
    #             'val_examples_per_class': 50
    #         },
    #         'standardize_graph': True
    #     }
    # }
    print('_config',_config)
    _config['data_path'] = generate_data_path(dataset,data_conf['dataset_source'])
    if _config['dataset_source'] == 'planetoid':
        print("dataset_source is planetoid")
        return get_dataset_and_split_planetoid(dataset, _config['data_path'])
    else:
        print("dataset_source is npz")
        adj, features, labels = get_dataset(dataset, _config['data_path'],
                                            _config['train_config']['standardize_graph'],
                                            _config['train_config']['split']['train_examples_per_class'],
                                            _config['train_config']['split']['val_examples_per_class'])
        random_state = np.random.RandomState(_config['seed'])

        print("adj.shape = ", adj.shape)
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels,
                                                                **_config['train_config']['split'])
        return adj, features, labels, idx_train, idx_val, idx_test

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c','--config-file',type=str,
                        default='configs/config.yml',
                        help='Path to the YAML configuration file for the experiment.')
    args = parser.parse_args()
    adj,features,labels,idx_train,idx_val,idx_test = load_dataset_and_split(args.config_file)
'''