import argparse

import numpy as np
import torch
import torch.nn as nn
#import visdom
from IPython.core.debugger import set_trace
import ramaPlot
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from utils.mnist_cached import MNISTCached as MNIST
#from utils.mnist_cached import setup_data_loaders
from top500 import setup_data_loaders
from utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples
import vonMises
#np.set_printoptions(threshold=np.inf)


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup three linear transformations (size of each input sample, size of each output sample)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linear transformation
        self.softplus = nn.Softplus()

    def forward(self, x):
        #set_trace()
        # define the forward computation on the angle-pair x
        # first shape the mini-batch to have angles in the rightmost dimension
        # TODO: Experiment with the shape. What shape does x have by default?
        x = x.reshape(-1, 2)
        # compute the hidden nodes
        # input x -> softplus the weights -> give that to the encoder hidden nodes
        hidden = self.softplus(self.fc1(x))
        # return a mean vector and a positive covariance
        mean = self.fc21(hidden)
        cvariance = torch.exp(self.fc22(hidden))
        #print("Encoder")
        #set_trace()
        return mean, cvariance

# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 2)
        self.fc22 = nn.Linear(hidden_dim, 2)
        # setup the non-linear transformation
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        # input z -> softplus the weights -> give that to the decoder hidden nodes
        hidden = self.softplus(self.fc1(z))
        # return the parameters for the output VonMises
        # TODO: play around with the mean and kappa. Perhaps use ReLu like the TorusDMM.
        mean = torch.sigmoid(self.fc21(hidden))*2*np.pi
        kappa = torch.sigmoid(self.fc22(hidden))*90 + 10
        #print("Decoder")
        #set_trace()
        return mean, kappa


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden nodes
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        self.z_dim = z_dim

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda

        # TODO: when I wake up continue and work on the model next

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module 'decoder' with Pyro
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            #print("Model")
            #set_trace()
            # setup hyperparameters for prior p(z)
            mean = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            cvariance = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            #set_trace()
            # sample from prior (value will be sampled by guide when computing ELBO)
            z = pyro.sample('latent', dist.Normal(mean, cvariance).to_event(1))
            # decode the latent z
            mean, kappa = self.decoder.forward(z)
            # TODO: play around with the reshape
            pyro.sample('obs', vonMises.VonMises(mean, kappa).to_event(1), obs=x.reshape(-1, 2))
            #set_trace()
            #print("Model")
            #set_trace()
            return mean, kappa

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            #print("Guide")
            #set_trace()
            # use the encoder to get the parameters used to define q(z|x)
            mean, cvariance = self.encoder.forward(x)
            #set_trace()
            # sample the latent z
            test = pyro.sample("latent", dist.Normal(mean, cvariance).to_event(1))
            #print("Guide")
            #set_trace()



    # define a helper function for reconstructing ramachandran plot
    def reconstruct_plot(self, x):
        # encode angles x
        #set_trace()
        mean, cvariance = self.encoder(x)
        # sample in latent space
        z = dist.Normal(mean, cvariance).sample()
        # decode the plot (note we don't sample in angle space)
        mean, kappa = self.decoder(z)
        return mean, kappa

    def sample_model(self):
        z = dist.Normal(0, 1).sample()
        mean, kappa = self.decoder(z)
        return mean, kappa

    #def sample(selfself, svi, ):


def main(args):
    # clear param store
    pyro.clear_param_store()

    # setup Protein data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=args.cuda)
    ramaPlot.plot_loader(train_loader)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)
    #set_trace()
    # setup optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    """
    for x, y in train_loader:
        print("x in train_loader:", x)
        set_trace()
        """

    train_elbo = []
    test_elbo = []

    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_loader:
            # if on GPU put mini-batch into CUDA memory
            # float() needed because it is currently double on that gives error
            # plus double datatype is considerably slower on GPU
            x = x.float()
            if args.cuda:
                x = x.float()
                x = x.cuda()
            # do ELBO gradient and accumulate los
            #set_trace()
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            for i, x in enumerate(test_loader):
                #set_trace()
                # float() needed because it is currently double on that gives error
                # plus double datatype is considerably slower on GPU
                x = x.float()
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

                # pick three random test angles from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    #print("i = 0")
                    #ramaPlot.plot_vae_samples(vae,str(epoch))
                    #ramaPlot.make_plots(train_elbo, train_loader, elbo, vae, x)
                    ramaPlot.make_plots2(train_elbo, train_loader, elbo, vae)
                    reco_indices = np.random.randint(0, x.shape[0], 3)
                    for index in reco_indices:
                        test_angle = x[index, :]
                        reco_angle = vae.reconstruct_plot(test_angle)
                        #set_trace()
                    #set_trace()
                    #ramaPlot.plot(test_angle, str(epoch)+"test_angle")
                    #ramaPlot.plot(reco_angle, str(epoch)+"reconstructed_angle")


            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))
        """
        if epoch == args.tsne_iter:
            mnist_test_tsne(vae=vae, test_loader=test_loader)
            plot_llk(np.array(train_elbo), np.array(test_elbo))
        """
    return vae


if __name__ == '__main__':
    # assert pyro.__version__.startswith('0.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10000, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    # parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)