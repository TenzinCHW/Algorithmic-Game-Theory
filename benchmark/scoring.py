import os
from os.path import join, expanduser
import numpy as np
import json
from benchmark.fid.fid_score import _compute_statistics_of_path, calculate_fid_given_paths
from benchmark.fid.inception import InceptionV3

BATCH_SIZE = 256
CUDA = True
DIMS = 2048

def save_ref_statistics(ref_images_path, ref_stats_dir):
    os.makedirs(ref_stats_dir, exist_ok=True)
    inception_model = init_inception()
    save_statistics(inception_model,
                    ref_images_path,
                    'results/ref/ref_fid.npz')

def save_test_statistics(model_dir):
    inception_model = init_inception()
    for run in os.listdir(model_dir):
        run_dir = join(model_dir, run)
        images_dir = join(run_dir, 'images')
        bench_dir = join(run_dir, 'bench')
        os.makedirs(bench_dir, exist_ok=True)
        for iteration in os.listdir(images_dir):
            save_statistics(inception_model,
                            join(images_dir,
                                 f'{iteration}'),
                            join(bench_dir,
                                 f'{iteration}.npz'))

def init_inception():
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
    model = InceptionV3([block_idx])
    if CUDA:
        model.cuda()
    return model.cuda()

def save_statistics(model, images_dir, save_path):
    mu, sigma = _compute_statistics_of_path(images_dir, model, BATCH_SIZE, DIMS, CUDA)
    np.savez(save_path, mu=mu, sigma=sigma)

def save_fid_scores(ref_stats_path, model_dir):
    fid_scores = {}
    for run in os.listdir(model_dir):
        test_stat_dir = join(model_dir, run, 'bench')
        for path in os.listdir(test_stat_dir):
            if 'npz' not in path:
                continue
            paths = (ref_stats_path, join(test_stat_dir, path))
            _, _, fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, CUDA, DIMS)
            fid_scores[path] = fid_score
        with open(join(test_stat_dir, 'fid.json'), 'w') as f:
            json.dump(fid_scores, f, indent=2)

