import os
from os.path import join, isfile, expanduser
from benchmark.scoring import *

ref_stat_dir = 'results/ref'
if not isfile(join(ref_stat_dir, 'ref_fid.npz')):
    save_ref_statistics(expanduser('~/Datasets/cifar-10/images'), ref_stat_dir)

for model_type in os.listdir('results'):
    if model_type == 'ref':
        continue
    model_dir = join('results', model_type)
    save_test_statistics(model_dir)

    ref_stat_path = join(ref_stat_dir, 'ref_fid.npz')
    save_fid_scores(ref_stat_path, model_dir)
    save_inception_scores(model_dir)

