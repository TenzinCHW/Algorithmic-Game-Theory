from model_utils.torchnn import *
import model_utils.opt as opt

def convxpose_bn_relu(inp_c, outp_c, kernel_pad):
    layer = [nn.ConvTranspose2d(inp_c, outp_c, *kernel_pad, bias=False),
            nn.BatchNorm2d(outp_c),
            nn.LeakyReLU(0.2, True)]
    return layer

def conv_bn_relu(inp_c, outp_c, kernel_pad):
    layer = [nn.Conv2d(inp_c, outp_c, *kernel_pad, bias=False),
            nn.BatchNorm2d(outp_c),
            nn.LeakyReLU(0.2, inplace=True)]
    return layer

def get_feat_channels(target_feat_channels):
    feat_channels = []
    start_c, end_c = target_feat_channels
    while start_c <= end_c / 2:
        feat_channels.append(int(start_c))
        start_c *= 2
    return feat_channels

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, target_feat_channels=(1,8)):
        super(Generator, self).__init__()
        feat_channels = get_feat_channels(target_feat_channels)
        feat_channels.reverse()
        convxpose_layers = [convxpose_bn_relu(opt.ngf * i * 2, opt.ngf * i, (4, 2, 1)) for i in feat_channels]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            *convxpose_bn_relu(opt.latent_dim, opt.ngf * 8, (4, 1, 0)),
            # state size. (ngf*8) x 4 x 4
            *[layer for layers in convxpose_layers for layer in layers],
            # state size. (ngf*4) x 8 x 8
            # state size. (ngf*2) x 16 x 16
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(-1).unsqueeze(-1)
        return self.main(inp)

class Discriminator(nn.Module):
    def __init__(self, target_feat_channels=(1,8)):
        super(Discriminator, self).__init__()
        feat_channels = get_feat_channels(target_feat_channels)
        conv_layers = [conv_bn_relu(opt.ndf * i, opt.ndf * i * 2, (4, 2, 1)) for i in feat_channels]
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.channels, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            *[layer for layers in conv_layers for layer in layers],
            # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            # state size. (ndf*8) x 4 x 4
        )

        self.feature_extractor = nn.Conv2d(opt.ndf * 8, opt.latent_dim, 4, 1, 0, bias=False)

        self.decoder = nn.Sequential(
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, inp):
        self.calc_grad(True)
        inp = self.encoder(inp)
        return self.decoder(inp)

    def get_features(self, inp):
        self.calc_grad(False)
        inp = self.encoder(inp)
        return self.feature_extractor(inp)

    def calc_grad(self, yes_or_no):
        for layer in self.encoder.parameters():
            layer.requires_grad = yes_or_no

def optimizer(net, lr):
    return optim.Adam(net.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))

criterion = nn.BCELoss()

def train_D(self):
    Tensor = torch.cuda.FloatTensor
    def train(real_imgs, cls):
        self.optimizer_D.zero_grad()
        batch_sz = real_imgs.shape[0]
        label = Tensor(np.ones((batch_sz,)))
        real_pred = self.D(real_imgs).view(-1)
        real_loss = criterion(real_pred, label)
        real_loss.backward()

        z = Tensor(np.random.normal(0, 1, (batch_sz, opt.latent_dim)))
        fake_imgs = self.G(z).detach()
        label = Tensor(np.zeros((batch_sz,)))
        fake_pred = self.D(fake_imgs).view(-1)
        fake_loss = criterion(fake_pred, label)
        fake_loss.backward()
        self.optimizer_D.step()
        loss_D = real_loss + fake_loss
        return loss_D.item(), z
    return train

def train_G(self):
    Tensor = torch.cuda.FloatTensor
    def train(z, real_imgs, cls):
        self.optimizer_G.zero_grad()
        batch_sz = real_imgs.shape[0]
        label = Tensor(np.ones((batch_sz,)))
        fake_imgs = self.G(z)
        fake_pred = self.D(fake_imgs).view(-1)
        loss_G = criterion(fake_pred, label)
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item(), fake_imgs
    return train

