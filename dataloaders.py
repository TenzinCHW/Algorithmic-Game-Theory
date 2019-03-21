import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from os.path import expanduser

image_size = 64
batch_size = 128
workers = 2

transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

home = expanduser('~')
data_root = f'{home}/Datasets'
train_set = dset.CIFAR10(root=data_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
test_set = dset.CIFAR10(root=data_root, train=False,
                                       transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
