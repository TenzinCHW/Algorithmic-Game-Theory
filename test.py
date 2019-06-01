import os
from os.path import join, isfile, expanduser
from benchmark.scoring import *

def test_model_type(model_type):
    model_dir = join('results', model_type)
    save_test_statistics(model_dir)
    ref_stat_path = join(ref_stat_dir, 'ref_fid.npz')
    save_fid_scores(ref_stat_path, model_dir)

    save_inception_scores(model_dir)

if __name__ == '__main__':
    ref_stat_dir = 'results/ref'
    if not isfile(join(ref_stat_dir, 'ref_fid.npz')):
        save_ref_statistics(expanduser('~/Datasets/cifar-10/images'), ref_stat_dir)

    to_test = ['Resnet', 'DCGANSpect', 'DCGANSpect_2N', 'DCGANSpect_Soft2N']
    for model_type in os.listdir('results'):
        if model_type == 'ref':
            continue
        if model_type == 'testdc':
            test_model_type(model_type)
        #if model_type in to_test:
        #    test_model_type(model_type)

