import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

def get_feat_channels(target_feat_channels):
    feat_channels = []
    start_c, end_c = target_feat_channels
    while start_c <= end_c / 2:
        feat_channels.append(int(start_c))
        start_c *= 2
    return feat_channels

