import pdb
import torch
from torch import nn as nn
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F

# using CNN to classify images
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

def cal_loss(model, eval_data, y ):

    pred = model(x)         # size: batch * 10
    # goal is to collect pred[0, y[0]], pred[1,y[1]],...pred[99, y[99]]
    select_pred = torch.gather(pred, dim = 1, index=y)        # bach * 1
    log_prob = torch.log(select_pred)
    loss = -log_prob.mean()
    return loss

def train():
    train_data = datasets.MNIST('../data', train=True, download=True)
    test_data = datasets.MNIST('../data', train=False, download=True)

    batch_size = 4

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                               shuffle = True, num_workers = 2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                               shuffle = True, num_workers = 2)
    #define a CNN:
    net = Net()

    # define a loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)



    for epoch in range(2):

        running_loss= 0.0
        for i, data in enumerate(train_loader, 0):

            print(i)
            pdb.set_trace()
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item

            if i % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == "__main__":
    ## TODO
    ## run the final function to classify the images here:
    #classify()
    train()
