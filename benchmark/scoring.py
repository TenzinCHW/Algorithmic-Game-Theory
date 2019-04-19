import os
from os.path import join
import numpy as np
from fid.fid_score import _compute_statistics_of_path

BATCH_SIZE = 256
CUDA = True
DIMS = 2048

def save_ref_statistics():
    save_test_statistics(expanduser('~/Datasets/cifar-10'), 'results/ref/ref.npz')

def save_test_statistics(test_path, epoch):
    save_statistics(join(test_path, 'images', f'{epoch}'), join(test_path, 'bench', f'{epoch}.npz'))

def save_statistics(path, save_path):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    mu, sigma = _compute_statistics_of_path(path, model, BATCH_SIZE, DIMS, CUDA)
    np.savez(save_path, mu=mu, sigma=sigma)

def save_fid_scores(ref_stat_path, test_path):
    stat_path = join(test_path, 'bench')
    for path in os.listdir(stat_path):
        paths = (ref_stat_path, join(stat_path, path))
        _, _, fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, CUDA, DIMS)
        np.savez(join(stat_path, f'fid{path}.npz'), )

