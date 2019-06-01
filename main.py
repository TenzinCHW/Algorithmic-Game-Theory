from Trainer import Trainer
from models import DCGAN, DCGAN_2N, DCWGAN_2N, SAGAN
from model_utils import opt
from model_utils.dataloaders import train_loader
from model_utils.torchnn import cudnn

cudnn.benchmark = True
models = [wgan, wgan_gp, DCGAN]

trainer = Trainer(DCWGAN_2N, train_loader, opt)
trainer.train_hyperparams()

#for model in models:
#    trainer = Trainer(model, train_loader, opt)
#    trainer.train_hyperparams()
#
