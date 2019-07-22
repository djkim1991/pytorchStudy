'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/6
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms


class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5)
        self.fc1 = nn.Linear(in_features=30*5*5, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, (2, 2))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

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