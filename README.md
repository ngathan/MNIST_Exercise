# MNIST_Exercise-

The task is to build a 2 layer MLP to classify images.  

Dataset: https://pytorch.org/vision/stable/datasets.html#mnist

Tools: Pytorch 

Install PyTorch as `pip3 install torch torchvision`

Use PyTorch to both download the dataset and write models for training. See this: https://github.com/pytorch/examples/blob/master/mnist/main.py#L116

The goal is to train a 2 layer MLP network with the following structure:

1. Flatten the images into a single feature
2. 1 Linear layer
3. 1 ReLu layer
4. 1 Linear layer to an output vector of size 10 (one for each digit)
5. At the end, a softmax layer (to generate probabilities over these 10 digits)

Use log-loss also called cross-entropy loss. Also, report accuracy where you predict the digit with the highest probability. Try to get 90%+ accuracy on the test set.

