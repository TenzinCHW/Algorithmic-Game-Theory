from Trainer import Trainer
from models import wgan, wgan_gp, DCGAN, SAGAN
from model_utils import opt
from model_utils.dataloaders import train_loader
from model_utils.torchnn import cudnn

cudnn.benchmark = True
trainer = Trainer(DCGAN, train_loader, opt)
trainer.train_hyperparams()

