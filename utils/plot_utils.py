import matplotlib.pyplot as plt
import re
import torch
import numpy as np
import copy

def parameters(model):
    num_params = 0
    for params in model.parameters():
        curn = 1
        for size in params.data.shape:
            curn *= size
        num_params += curn
    return num_params

class statistics_feature():
    """
    get the statistics of the feature
    """
    def __init__(self):
        self.feat = None
        
    def _min(self):
        self.min = np.min(self.feat)

    def _max(self):
        self.max = np.max(self.feat)

    def _hist(self, _min, _max):
        hist, _ = np.histogram(self.feat, bins=100, range=(_min, _max))
        self.hist = hist

    def add(self, feat):
        feat = copy.deepcopy(feat)
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        self.feat = feat
    
    def plot(self, saveto=None):
        self._max()
        self._min()
        self._hist(_min=-1, _max=1)
        plt.plot(self.hist)
        if saveto is None:
            print("min and max: ", self.min, self.max)
            plt.show()


class loss_logger():
    """class for logging the loss
    """
    def __init__(self):
        self.lossdict = {}
        self.x = 1
    
    def add(self, name, score, x=None):
        if x is None:
            x = self.x; self.x+=1

        if name in self.lossdict:
            self.lossdict[name] += [(x,score)]
        else:
            self.lossdict[name] = [(x,score)]

    def plot(self, names=None, saveto=None):
        if names is None:
            names = self.lossdict.keys()
        for name in names:
            x = [data[0] for data in self.lossdict[name]]
            y = [data[1] for data in self.lossdict[name]]
            plt.plot(x, y)
        plt.legend(names)
        if saveto is not None:
            plt.savefig(saveto)
        else:
            plt.show()
        plt.close()


def read_log(filepath):
    train = []
    val = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        info = re.findall(r'\d+',line)
        if len(info)==3:
            value = float(info[2])*1e-4
            if "train" in line:
                train.append(value)
            elif "val" in line:
                val.append(value)
    plt.plot(train)
    plt.plot(val)

if __name__ == '__main__':

    losslogger = loss_logger()
    losslogger.add('g', 3)
    losslogger.add('g', 4)
    losslogger.add('g', 5)
    losslogger.add('g', 6)
    losslogger.add('g', 3)
    losslogger.plot()
