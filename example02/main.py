'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/2
'''

import torch
import numpy as np


class Example:
    # Auto gradient and Variable
    @staticmethod
    def auto_grad_and_variable():
        a = torch.Tensor(np.array([[1]]))
        a = torch.autograd.Variable(a, requires_grad=True)

        b = a + 2
        c = b**2
        d = c*3

        d.backward()

        print("="*50)
        print("d.data:", d.data)
        print("d.grad:", d.grad)
        print("d.grad_fn:", d.grad_fn)

        print("="*50)
        print("c.data:", c.data)
        print("c.grad:", c.grad)
        print("c.grad_fn:", c.grad_fn)

        print("="*50)
        print("b.data:", b.data)
        print("b.grad:", b.grad)
        print("b.grad_fn:", b.grad_fn)

        print("="*50)
        print("a.data:", a.data)
        print("a.grad:", a.grad)
        print("a.grad_fn:", a.grad_fn)