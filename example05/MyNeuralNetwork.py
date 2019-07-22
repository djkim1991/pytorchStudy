'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/5
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms


class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.net_1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.net_2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5)

    def forward(self, x):
        x = self.net_1(x)
        x = self.net_2(x)

        return x

    @staticmethod
    def load_data():
        # classes of "CIFAR10"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transform
        )

        test_set = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=False,
            transform=transform
        )
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)

        return train_loader, test_loader