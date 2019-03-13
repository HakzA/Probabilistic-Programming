import os
from VAE import VAE
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from trainEval import train, evaluate
import torch
from loader import getData
import pyro
from IPython.core.debugger import set_trace
from dataLoader import setup_data_loaders
import argparse

import numpy as np
import torch
import torch.nn as nn

from IPython.core.debugger import set_trace

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


"""""
# Check version and more
assert pyro.__version__.startswith('0.3.0')
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ
"""""

def main(args):
    # clear param store
    pyro.clear_param_store()

    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(use_cuda=args.cuda, batch_size=256)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    #set_trace()
    # setup visdom for visualization


    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
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
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them


            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

        if epoch == args.tsne_iter:
            mnist_test_tsne(vae=vae, test_loader=test_loader)
            plot_llk(np.array(train_elbo), np.array(test_elbo))

    return vae


if __name__ == '__main__':
    #assert pyro.__version__.startswith('0.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)

"""""
# Run options
LEARNING_RATE = 1.0e-3
USE_CUDA = False

# Run only for a single iteration for testing
NUM_EPOCHS = 1 if smoke_test else 100
TEST_FREQUENCY = 5

# Get data
train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)
pyro.clear_param_store()

# Initialize instance of the VAE class
vae = VAE()

# Setup Adam optimizer (an algorithm for first-order gradient-based optimization)
optimizer = Adam({"lr": 1.0e-3})

# SVI: stochastic variational inference - a scalable algorithm for approximating posterior distributions.
# Trace_ELBO: top-level interface for stochastic variational inference via optimization of the evidence lower bound.
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())


#set_trace()

train_elbo = []
test_elbo = []

# Training loop
for epoch in range(NUM_EPOCHS):
    print("1")
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
    #print("2")
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
"""""