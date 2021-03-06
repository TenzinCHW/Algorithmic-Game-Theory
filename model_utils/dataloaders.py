import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from os.path import expanduser
from model_utils.opt import img_size, batch_size, n_cpu

transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

home = expanduser('~')
data_root = f'{home}/Datasets/cifar-10'
train_set = dset.CIFAR10(root=data_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True, num_workers=n_cpu)
test_set = dset.CIFAR10(root=data_root, train=False,
                                       transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=n_cpu)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
