from os import listdir
from os.path import join
import json

def loop_path(path):
    files = listdir(path)
    return [join(path, f) for f in files]

def replace_all_npz(base_path):
    for model_type in loop_path(base_path):
        if model_type == 'results/ref':
            continue
        for run in loop_path(model_type):
            replace_fid_file(join(run, 'bench', 'fid.json'))

def replace_fid_file(fid_file_name):
    new_fid_scores = replace_fid_npz(fid_file_name)
    with open(fid_file_name, 'w') as f:
        json.dump(new_fid_scores, f, indent=2)

def replace_fid_npz(fid_file_name):
    fid_scores = get_fid_content(fid_file_name)
    new_fid_scores = {}
    for k, v in fid_scores.items():
        new_fid_scores[replace_npz(k)] = v
    return new_fid_scores

def get_fid_content(fid_file_name):
    with open(fid_file_name, 'r') as f:
        content = json.load(f)
    return content

def replace_npz(name):
    return name.split('.')[0]

if __name__ == '__main__':
    print(replace_npz('1000.npz'))
    print(loop_path('results'))
    print(get_fid_content('results/wgan/0.002.5/bench/fid.json'))
    print(replace_all_npz('results'))

