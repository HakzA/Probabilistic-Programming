from __future__ import print_function
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import loader
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from IPython.core.debugger import set_trace



def load(train=True):
    """
    Loads dataset.
    :param train: A bool indicating which dataset to load
    :return: train set = True and test set = False
    """

    # Loading the protein.pkl file into a Python object
    dataset = np.array(loader.load_data())

    # The dataset has the structure dataset[[training set], [test set]]
    if train:
        dataset = dataset[0]
    else:
        dataset = dataset[1]

    return dataset


class top500(data.Dataset):
    """`
    top500 protein Dataset.
    :param train: A bool indicating which dataset to load

    """

    def __init__(self, train=True):
        # Either the train or test set
        self.data = load(train=train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (phi, psi) angle pairs at the given index.
        """
        anglepair = self.data[index]
        return anglepair

    def __len__(self):
        """
        :return: Length of the dataset
        """
        return len(self.data)

'''
sampler = RandomSampler(top500())
iterator = sampler.__iter__()
for i in iterator:
    print(i)
    next(iterator)
'''



def setup_data_loaders(batch_size=128, use_cuda=False):
    """
            helper function for setting up pytorch data loaders for a semi-supervised dataset
        :param dataset: the data to use
        :param use_cuda: use GPU(s) for training
        :param batch_size: size of a batch of data to output when iterating over the data loaders
        :param kwargs: other params for the pytorch data loader
        :return: three data loaders: (supervised data for training, un-supervised data for training,
                                      supervised data for testing)
    """

    # Default number of workers and use_cuda
    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}

    train_set = top500(train=True)
    test_set = top500(train=False)

    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader




# load()
'''
dataset = top500(train=False)

print(dataset.__getitem__(0))
print(dataset.__len__())
print(dataset.data)
'''