'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/7
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms


class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=128, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        x = self.layer4(x)

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