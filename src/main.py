import pdb
import torch
from torch import nn as nn
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    train_data = datasets.MNIST('../data', train=True, download=True)
    test_data = datasets.MNIST('../data', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(train_data)
    test_loader = torch.utils.data.DataLoader(test_data)

    pdb.set_trace()

    #define a CNN:
    net = Net()

    # define a loss function and optimizer

    criterion = nn.CrossntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)


    for epoch in range(2):

        running_loss= 0.0
        #for i, data in enumerate(train_loader, 0):






# create helper functions here
# using CNN to classify images
#def :


if __name__ == "__main__":
    ## TODO
    ## run the final function to classify the images here:
    #classify()
    main()
