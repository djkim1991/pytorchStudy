'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/4
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Example:

    # conv2d, example of torch.nn.functional
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

        # print shapes of loaded data
        for n, (img, labels) in enumerate(test_loader):
            print(n, img.shape, labels.shape)

        # show data's image in first batch
        test_iter = iter(test_loader)
        images, labels = test_iter.next()
        Example.imshow(torchvision.utils.make_grid(images, nrow=4))
        for label in labels:
            print(classes[label])

    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5
        np_img = img.numpy()    # C*H*W -> H*W*C
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

        print(np_img.shape)
        print(np.transpose(np_img, (1, 2, 0)).shape)


if __name__ == '__main__':
    Example.load_data()
