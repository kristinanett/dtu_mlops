import argparse
import sys

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from data import mnist
from model import Net


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
        
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        num_classes = 10
        model = Net(num_classes)
        trainloader, _ = mnist()
        x, y = next(iter(trainloader))
        print("Batch dimension [B x C x H x W]:", x.shape)

        num_epoch = 3
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        losses = []

        # training loop
        for epoch in range(num_epoch):  
            running_loss = 0.0
            model.train()

            for i, data in enumerate(trainloader, 0):
                # get the inputs (data is a list of [inputs, labels])
                inputs, labels = data 

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels.long())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)

                # compute gradients given loss
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
  
                # print statistics
                running_loss += loss.item() #loss.data[0]

                if i % 1000 == 999:    # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                    losses.append(running_loss / 1000) #(loss.data.numpy())
                    running_loss = 0.0

        torch.save(model.state_dict(), 'trained_model.pt')
        print('Finished Training')

        plt.figure(figsize=(9, 9))
        plt.plot(np.array(losses), label='Training Error')
        plt.legend(fontsize=20)
        plt.xlabel('Train step', fontsize=20)
        plt.ylabel('Error', fontsize=20)
        plt.savefig('trainerror.png')

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        num_classes = 10
        model = Net(num_classes)
        state_dict  = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)

        _, testloader = mnist()

        correct = 0
        total = 0

        for data in testloader:
            images, labels = data
            images, labels = Variable(images), Variable(labels.long())

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the test images: {:4.2f} %'.format(100 * np.true_divide(correct.numpy(), total)))


if __name__ == '__main__':

    TrainOREvaluate()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    