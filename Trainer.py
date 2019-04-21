import os
import json
from itertools import product
from collections import namedtuple, defaultdict
from model_utils.torchnn import torch, vutils, cudnn
from os.path import join
from tqdm import tqdm

def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

class Trainer:
    def __init__(self, GAN_type, dataloader, opt, custom_name=''):
        self.GAN_type = GAN_type
        self.model_type_name = self.GAN_type.__spec__.name.split('.')[-1] # We only want the name of the module
        self.loader = dataloader
        self.opt = opt
        self.device = torch.device('cuda:0')
        self.train_D = self.GAN_type.train_D(self)
        self.train_G = self.GAN_type.train_G(self)
        self.custom_name = custom_name

    def train_hyperparams(self, continue_from=0):
        assert(continue_from >= 0)
        for params in dict_product(self.opt.hyperparams):
            hyperparam_vals = [str(val) for val in params.values()]
            path = f'results/{self.model_type_name+self.custom_name}/{".".join(hyperparam_vals)}'
            for save_type in ['images', 'model']:
                os.makedirs(join(path, save_type), exist_ok=True)
            hyperparams = namedtuple('Hyperparameters', params.keys())(**params)
            losses_path = join(path, 'losses.json')
            self.D = self.GAN_type.Discriminator().to(self.device)
            self.G = self.GAN_type.Generator().to(self.device)
            if continue_from > 0:
                self.D.load_state_dict(torch.load(join(path, 'model', f'D{continue_from}.pth')))
                self.G.load_state_dict(torch.load(join(path, 'model', f'G{continue_from}.pth')))
                with open(losses_path, 'r') as f:
                    prev_losses = json.load(f)
            losses = self.train(hyperparams, path, start=continue_from)
            if continue_from > 0:
                losses.update(prev_losses)
            with open(losses_path, 'w') as f:
                json.dump(losses, f, indent=2)

    def train(self, hyperparams, path, start):
        self.optimizer_D = self.GAN_type.optimizer(self.D, hyperparams.lr)
        self.optimizer_G = self.GAN_type.optimizer(self.G, hyperparams.lr)
        losses = dict()

        num_batches = 0
        for epoch in range(start, self.opt.n_epochs):
            average_D_loss = 0
            average_G_loss = 0
            epoch_losses = defaultdict(lambda: [])
            for i, (imgs, cls) in tqdm(enumerate(self.loader)):
                imgs = imgs.to(self.device)
                D_loss, z = self.train_D(imgs, cls)
                average_D_loss += (average_G_loss * i + D_loss) / (i + 1)
                epoch_losses['D'].append(D_loss)
                epoch_losses['ave_D'].append(average_D_loss)

                if i % hyperparams.n_critic == 0:
                    G_loss, fake_imgs = self.train_G(z, imgs, cls)
                    average_G_loss += (average_G_loss * i / hyperparams.n_critic + G_loss) / (i / hyperparams.n_critic + 1)
                    epoch_losses['G'].append(G_loss)
                    epoch_losses['ave_G'].append(average_G_loss)

                if num_batches % self.opt.sample_interval == 0:
                    loc = join(path, f'images/{num_batches}')
                    os.makedirs(loc)
                    for i, img in enumerate(fake_imgs[:]):
                        vutils.save_image(img, f'{loc}/{i}.png')
                num_batches += 1
            torch.save(self.D.state_dict(), join(path, f'model/D{epoch}.pth'))
            torch.save(self.G.state_dict(), join(path, f'model/G{epoch}.pth'))
            losses[epoch] = epoch_losses
        return losses

