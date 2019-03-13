import pyro.distributions.torch_distribution.TorchDistribution as pd
import torch

def main():
    vm = pd.VonMises(0, 5)
    print(vm)

main()

