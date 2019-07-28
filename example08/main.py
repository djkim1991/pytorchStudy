'''
    writer: dororongju
    github: https://github.com/djkim1991/pytorchStudy/issues/8
'''
from example07.MyNeuralNetwork import MyNeuralNetwork
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

class Example:
    @staticmethod
    def save_model():
        network = MyNeuralNetwork()
        train_loader, test_loader = network.load_data()

        optimizer = torch.optim.SGD(params=network.parameters(), lr=0.001, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()

        if os.path.isfile('cnn.pth'):
            # load model
            network.load_state_dict(torch.load('./cnn.pth'))

        epoch_size = 3
        for epoch in range(epoch_size):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    network.cuda()

                optimizer.zero_grad()
                out = network.forward(inputs)
                loss = loss_function(out, labels)
                loss.backward()
                optimizer.step()

                if(i % 100 == 0):
                    print('{0}: loss is {1}'.format(i, loss))

        print("train over")

        # save model
        torch.save(network.state_dict(), './cnn.pth')

        total = 0
        correct = 0
        for _, data in enumerate(test_loader):
            images, labels = data

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            out = network.forward(Variable(images))
            _, predicted = torch.max(out.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy is {0}%'.format(100*correct/total))


if __name__ == '__main__':
    Example.save_model()
