import matplotlib.pyplot as plt
from os.path import join
from os import listdir
from model_utils import opt
import json
from collections import defaultdict

def extract_loss_by_mini_batch(raw_losses, by='batches'):
    Dmini_batches, Gmini_batches, D_loss, G_loss = [], [], [], []
    for i in range(opt.n_epochs):
        Dit, D, Git, G = extract_epoch(raw_losses, i)
        Dmini_batches.extend(Dit)
        Gmini_batches.extend(Git)
        D_loss.extend(D)
        G_loss.extend(G)
    return {f'Number of {by}' : Dmini_batches,
            'Discriminator' : D_loss}, \
            {f'Number of {by}' : Gmini_batches,
            'Generator' : G_loss}

def extract_epoch(raw_losses, i):
    epoch_loss = raw_losses[str(i)]
    D_loss, G_loss = epoch_loss['D'], epoch_loss['G']
    Dmini_batches_per_epoch = len(D_loss)
    Gmini_batches_per_epoch = len(G_loss)
    Dmini_batches = [i * Dmini_batches_per_epoch + j for j in range(Dmini_batches_per_epoch)]
    Gmini_batches = [i * Gmini_batches_per_epoch + j for j in range(Gmini_batches_per_epoch)]
    return  Dmini_batches, D_loss, Gmini_batches, G_loss

def extract_inception(raw_inception, by='batches'):
    mini_batches = ext_keys(raw_inception)
    inception = [raw_inception[str(i)]['mu'] for i in mini_batches]
    return {f'Number of {by}' : mini_batches,
            'Inception Score' : inception}

def extract_fid(raw_fid, by='batches'):
    mini_batches = ext_keys(raw_fid)
    fid = [raw_fid[str(i)] for i in mini_batches]
    return {f'Number of {by}' : mini_batches,
            'Frechet Inception Score' : fid}

def ext_keys(score):
    return sorted(map(lambda i: int(i), score.keys()))

def plot(losses, item_to_plot, y_label='Loss', x_label='Minibatch'):
    for name, loss in losses.items():
        plot_item(name, loss, item_to_plot)
    plt.title(f'{item_to_plot} {y_label}')
    plt.ylabel(f'{y_label}')
    plt.xlabel(f'{x_label}')
    plt.legend()
    plt.show()

def plot_item(name, loss, item_to_plot):
    plt.plot(loss['Number of batches'], loss[item_to_plot], label=name)


D_losses, G_losses = {}, {}
inception_scores, fid_scores = {}, {}

for arch in ['DCGAN', 'DCGAN_2N', 'DCWGAN_2N', 'SAGAN']:
    base_path = join('results', arch, '0.0002.5')
    loss_path = join(base_path, 'losses.json')
    if arch == 'SAGAN':
        pass
    else:
        with open(loss_path) as f:
            raw_loss = json.load(f)
            D_losses[arch], G_losses[arch] = extract_loss_by_mini_batch(raw_loss)

    inception_path = join(base_path, 'bench', 'inception.json')
    with open(inception_path) as f:
        raw_inception_score = json.load(f)
        inception_scores[arch] = extract_inception(raw_inception_score)

    fid_path = join(base_path, 'bench', 'fid.json')
    with open(fid_path) as f:
        raw_fid = json.load(f)
        fid_scores[arch] = extract_fid(raw_fid)

plot(D_losses, 'Discriminator')
plot(G_losses, 'Generator')

plot(inception_scores, 'Inception Score', y_label='Score')
plot(fid_scores, 'Frechet Inception Score', y_label='Score')

for arch in ['DCGANSpect', 'DCGANSpect_2N', 'DCGANSpect_Soft2N']:
    pass

