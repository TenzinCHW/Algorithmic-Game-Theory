import os
from itertools import product
from collections import namedtuple
from model_utils.torchnn import torch, vutils, cudnn
from os.path import join
from tqdm import tqdm

def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

class Trainer:
    def __init__(self, GAN_type, dataloader, opt):
        self.GAN_type = GAN_type
        self.model_type_name = self.GAN_type.__spec__.name.split('.')[-1] # We only want the name of the module
        self.loader = dataloader
        self.opt = opt
        self.device = torch.device('cuda:0')
        self.train_D = self.GAN_type.train_D(self)
        self.train_G = self.GAN_type.train_G(self)

    def train_hyperparams(self):
        for params in dict_product(self.opt.hyperparams):
            hyperparam_vals = [str(val) for val in params.values()]
            path = f'results/{self.model_type_name}/{".".join(hyperparam_vals)}'
            for save_type in ['images', 'model']:
                os.makedirs(join(path, save_type))
            hyperparams = namedtuple('Hyperparameters', params.keys())(**params)
            self.train(hyperparams, path)

    def train(self, hyperparams, path):
        self.D = self.GAN_type.Discriminator().to(self.device)
        self.G = self.GAN_type.Generator().to(self.device)
        self.optimizer_D = self.GAN_type.optimizer(self.D, hyperparams.lr)
        self.optimizer_G = self.GAN_type.optimizer(self.G, hyperparams.lr)

        num_batches = 0
        for epoch in range(self.opt.n_epochs):
            for i, (imgs, cls) in tqdm(enumerate(self.loader)):
                imgs = imgs.to(self.device)
                D_loss, z = self.train_D(imgs, cls)

                if i % hyperparams.n_critic == 0:
                    loss_G, fake_imgs = self.train_G(z, imgs, cls)
                if num_batches % self.opt.sample_interval == 0:
                    loc = join(path, f'images/{num_batches}')
                    os.makedirs(loc)
                    for i, img in enumerate(fake_imgs[:]):
                        vutils.save_image(img, f'{loc}/{i}.png')
                num_batches += 1
            torch.save(self.D.state_dict(), join(path, f'model/{epoch}.pth'))

