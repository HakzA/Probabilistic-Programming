import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from dataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader



def setup_data_loaders(batch_size=128, use_cuda=False):
    train_set = ProteinDataset()
    test_set = ProteinDataset()
    #print("dataloader1")
    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, **kwargs)
    #print("daatloader")
    return train_loader, test_loader


#setup_data_loaders()

"""""
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    print(type(train_set))

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
"""""