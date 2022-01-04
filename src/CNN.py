import torch
from torch import nn as nn
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):  # noqa

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def im_show(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def train():

    batch_size = 4
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])])  # transform for black/ white images
    train_data = datasets.MNIST('../data', train=True,
                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    # define a CNN:
    net = Net()

    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # save training model
    path = './models/mnist_train.pth'
    torch.save(net.state_dict(), path)
    print("Finished Training")


def test():
    path = './models/mnist_train.pth'
    net = Net()
    net.load_state_dict(torch.load(path))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])  # transform for black/ white images

    batch_size = 4
    test_data = datasets.MNIST('../data', train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on the test set:', 100 * correct / total)


if __name__ == "__main__":
    train()
    test()
