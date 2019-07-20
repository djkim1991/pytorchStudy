'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/3
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Example:

    input_data = torch.Tensor(np.array([[[
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ]]]))

    # conv2d, example of torch.nn.functional
    @staticmethod
    def funational_conv2d():

        conv2d_filter = torch.Tensor(np.array([[[
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]]]))

        input_data = Variable(Example.input_data, requires_grad=True)
        conv2d_filter = Variable(conv2d_filter)

        out = F.conv2d(input_data, conv2d_filter)
        print(out)

    # conv2d, example of torch.nn
    @staticmethod
    def conv2d():
        input_data = Variable(Example.input_data, requires_grad=True)

        func = nn.Conv2d(1, 1, 3)
        print(func.weight)

        out = func(input_data)
        print(out)


if __name__ == '__main__':
    Example.functional_conv2d()
    Example.conv2d()
