from model_utils.torchnn import *
from model_utils import opt
from models import DCGAN

class Generator(DCGAN.Generator):
    def __init__(self):
        super(Generator, self).__init__()

class Discriminator(DCGAN.Discriminator):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = 2 * num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(opt.ndf * 8, self.num_classes, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(DCGAN.weights_init)

optimizer = DCGAN.optimizer

def train_D(self):
    def train(real_imgs, cls):
        num_cls = self.D.num_classes
        batch_sz = real_imgs.shape[0]
        self.optimizer_D.zero_grad()
        label = one_hot_embedding(cls, num_cls)
        real_pred = self.D(real_imgs).view(batch_sz, -1)
        real_loss = DCGAN.criterion(real_pred, label)
        real_loss.backward()

        z = self.D.get_features(real_imgs)
        fake_imgs = self.G(z).detach()
        label = one_hot_embedding(cls + num_cls/2, num_cls)
        fake_pred = self.D(fake_imgs).view(batch_sz, -1)
        fake_loss = DCGAN.criterion(fake_pred, label)
        fake_loss.backward()
        self.optimizer_D.step()
        loss_D = real_loss + fake_loss

        return loss_D.item(), z
    return train

def train_G(self):
    def train(z, real_imgs, cls):
        num_cls = self.D.num_classes
        batch_sz = real_imgs.shape[0]
        self.optimizer_G.zero_grad()
        fake_imgs = self.G(z)
        label = one_hot_embedding(cls, num_cls)
        pred = self.D(fake_imgs).view(batch_sz, -1)
        loss_G = DCGAN.criterion(pred, label)
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item(), fake_imgs
    return train

def one_hot_embedding(labels, num_cls):
    y = torch.eye(num_cls)
    labels = torch.LongTensor(labels)
    return y[labels].cuda()
