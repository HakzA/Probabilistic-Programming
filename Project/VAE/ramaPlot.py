import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
from matplotlib.colors import LogNorm
import loader
import torch
import numpy as np
from IPython.core.debugger import set_trace
import vonMises
from matplotlib import colors


def plot(angles, epoch):
    phi = [a[0] for a in angles]
    psi = [a[1] for a in angles]

    plt.hist2d(phi, psi, bins=628, norm=LogNorm())
    # block=False such that the program can continue running (else we would have to close the window first)
    # plt.show(block=False)
    # or we can just save the image instead...
    plt.savefig('ramaPlot_epoch_'+epoch+'.png')


def plot_samples(angles, epoch):
    # The shape of the array is different so the indexing has to be like this
    phi = [a[0][0] for a in angles]
    psi = [a[0][1] for a in angles]

    plt.hist2d(phi, psi, bins=628, norm=LogNorm())
    # block=False such that the program can continue running (else we would have to close the window first)
    # plt.show(block=False)
    # or we can just save the image instead...
    plt.savefig('ramaPlot_epoch_'+epoch+'.png')

def vae_samples(vae):
    x = torch.zeros([1, 2])
    output_angles = []
    output_means = []
    output_kappas = []
    for i in range(10):
        angles = []
        angle_mean = []
        angle_kappa = []
        for rr in range(100):
            sample_mean_i, sample_kappa_i = vae.model(x)
            sample_means_phi = sample_mean_i[0][0].detach() # Don't detach yet maybe, because when given to vonmises it want a tensor or number
            sample_means_psi = sample_mean_i[0][1].detach()
            sample_kappas_phi = sample_kappa_i[0][0].detach()
            sample_kappas_psi = sample_kappa_i[0][1].detach()
            #set_trace()
            phi = np.random.vonmises(sample_means_phi - np.pi, sample_kappas_phi)+ np.pi
            psi = np.random.vonmises(sample_means_psi - np.pi, sample_kappas_psi)+ np.pi

            angle_mean.append([sample_means_phi, sample_means_psi])
            angle_kappa.append([sample_kappas_phi, sample_kappas_psi])
            angles.append([phi, psi])

        output_angles.append(angles)
        output_means.append(angle_mean)
        output_kappas.append(angle_kappa)


            #angles.append(sample_mean_i.detach().numpy())
            #set_trace()
        #plot_samples(np.asarray(angles), epoch + "_samples")
        #set_trace()

    return np.array(output_angles), np.array(output_means), np.array(output_kappas)



# epoch_list: list of epoch loss (train_loss)
# training_seq_lengths: data['train']['sequences']
# elbo JitTrace_ELBO() if args.jit else Trace_ELBO()
def make_plots(vae, train_elbo):
    # epoch number
    n_iter = len(train_elbo)
    x_train = np.array(range(len(train_elbo)))
    plt.title('- ELBO')
    plt.xlabel('number of epochs')
    plt.plot(x_train, train_elbo)
    elbo_fn = "ELBO_" + str(n_iter) + ".png"
    plt.savefig(elbo_fn)
    # A BUNCH OF PLOTTING AND SAVING
    #
    #
    samples_angles, mean_angles, k_angles = vae_samples(vae)

    s_phi = []
    s_psi = []
    m_phi = []
    m_psi = []
    k_phi = []
    k_psi = []

    for i in range(len(samples_angles)):
        # print(samples_angles[i])
        s_i = np.array(samples_angles[i])
        m_i = np.array(mean_angles[i])
        k_i = np.array(k_angles[i])
        s_phi = s_phi + s_i[:, 0].flatten().tolist()
        s_psi = s_psi + s_i[:, 1].flatten().tolist()

        m_phi = m_phi + m_i[:, 0].flatten().tolist()
        m_psi = m_psi + m_i[:, 1].flatten().tolist()

        k_phi = k_phi + k_i[:, 0].flatten().tolist()
        k_psi = k_psi + k_i[:, 1].flatten().tolist()

    n_bins = 60
    axes = [[0, 2 * np.pi], [0, 2 * np.pi]]

    # print(len(s_psi), len(s_phi), len(s_psi), len(m_phi))
    # make ramachandran plot
    # clearf figure, make plots

    plt.clf()
    plt.title('ramachandran sampled angles')
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.hist2d(s_phi, s_psi, bins=n_bins, range=axes, norm=colors.LogNorm())
    RamSam_fn = "RamaSample_" + str(n_iter) + ".png"
    plt.savefig(RamSam_fn)

    plt.clf()
    plt.title('angle means')
    plt.xlabel('phi')
    plt.ylabel('psi')
    # plt.vlines(1.8, 0, 2*np.pi)
    # plt.hlines(3.8, 0, 2*np.pi)
    plt.hist2d(m_phi, m_psi, bins=n_bins, range=axes, norm=colors.LogNorm())
    Mean_fn = "Means_" + str(n_iter) + ".png"
    plt.savefig(Mean_fn)

    plt.clf()
    plt.title('angle ks')
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.hist2d(k_phi, k_psi, bins=n_bins, range=axes, norm=colors.LogNorm())
    k_fn = "Kappas_" + str(n_iter) + ".png"
    plt.savefig(k_fn)

    plt.clf()
    #dmm.cuda()

    return








#plot(loader.getData("top500.txt"))