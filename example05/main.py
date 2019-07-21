'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/5
'''
from example05.MyNeuralNetwork import MyNeuralNetwork
from torch.autograd import Variable


class Example:
    @staticmethod
    def forward_example():
        network = MyNeuralNetwork()
        train_loader, test_loader = network.load_data()

        test_iter = iter(test_loader)
        images, labels = test_iter.next()

        out = network.forward(Variable(images))

        print(out.shape)


if __name__ == '__main__':
    Example.myNeuralNetwork()
