import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import loader

def plot(angles):
    phi = [a[0] for a in angles]
    psi = [a[1] for a in angles]

    plt.hist2d(phi, psi, bins=628, norm=LogNorm())
    plt.show()



#plot(loader.getData("top500.txt"))