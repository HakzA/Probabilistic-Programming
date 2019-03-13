import torch
import numpy as np
from torch.autograd import Variable
from loader import getData
from torch.utils.data import Dataset, DataLoader
from IPython.core.debugger import set_trace
import torch.nn as nn

class ProteinDataset (Dataset):
    """ Protein dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        angles = getData("top500.txt")
        self.len = angles.shape[0]
        self.phi_data = torch.from_numpy(angles[:, 0:-1])
        self.psi_data = torch.from_numpy(angles[:, [-1]])

    # Return one item on the index
    def __getitem__(self, index):
        return self.phi_data[index], self.psi_data[index]

    # Return the data length
    def __len__(self):
        return self.len



class Encoder(nn.Module):
    """
    (Guide)
    We take our angles x, and encode them as z.
    We allow for each x_i to depend on z_i in a non-linear way.
    The dependency will be parameterized by a nn.

    z_dim:          The dimension of our latent space.
    hidden_dim:     Number of hidden units.
    x:

    returns:        A mean and variance for the normal dist. (from which we will sample z).
    """

    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the angle-pair x
        # compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale








def main():
    dataset = Dataset()
    train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

