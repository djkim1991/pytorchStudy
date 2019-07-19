'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/1
'''

import torch
import numpy as np


class Example:
    # introduce Pytorch
    @staticmethod
    def introduce():
        # define Tensor
        x = torch.Tensor(3)
        print(x)

        x = torch.Tensor(2, 5)
        print(x)

        # get random number(uniform distribution random)
        x = torch.rand(3, 3)
        print(x)

        # get random number(normal distribution)
        x = torch.randn(3, 3)
        print(x)

        # Numpy To Tensor
        x = np.array([1, 2, 3, 4])
        x = torch.Tensor(x)
        print(x)

        # Tensor To Numpy
        x = torch.rand(2, 4)
        x = x.numpy()
        print(x)

        # reshape Tensor
        x = torch.rand(1, 6)
        x = x.view(2, 3)
        print(x)

        # combine Tensors
        x = torch.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        y = torch.Tensor(np.array([[10, 20, 30], [40, 50, 60]]))
        z = torch.cat((x, y), dim=0)
        print(z)
        z = torch.cat((x, y), dim=1)
        print(z)

        # calculate by GPU
        x = torch.Tensor(np.array([1, 2]))
        y = torch.Tensor(np.array([10, 20]))

        if torch.cuda.is_available():
            print("using GPU")
            x = x.cuda()
            y = y.cuda()

        print(x + y)


if __name__ == '__main__':
    Example.introduce()