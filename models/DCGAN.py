from model_utils.torchnn import *
from model_utils.opt import *

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

class ConvGenerator(nn.Module):
    def __init__(self, target_feat_channels):
        super(ConvGenerator, self).__init__()
        feat_channels = get_feat_channels(target_feat_channels)
        convxpose_layers = [convxpose_bn_relu(ngf * i * 2, ngf * i, (4, 2, 1)) for i in feat_channels]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            *convxpose_bn_relu(nz, ngf * 8, (4, 1, 0)),
            # state size. (ngf*8) x 4 x 4
            *[layer for layers in convxpose_layers for layer in layers],
            # state size. (ngf*4) x 8 x 8
            # state size. (ngf*2) x 16 x 16
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inp):
        return self.main(inp)

class ConvDiscriminator(nn.Module):
    def __init__(self, target_feat_channels):
        super(ConvDiscriminator, self).__init__()
        feat_channels = get_feat_channels(target_feat_channels)
        conv_layers = [conv_bn_relu(ndf * i, ndf * i * 2, (4, 2, 1)) for i in feat_channels]
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            *[layer for layers in conv_layers for layer in layers],
            # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            # state size. (ndf*8) x 4 x 4
        )

        self.feature_extractor = nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False)

        self.decoder = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

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

