import torch
import torch.nn as nn
import torch.nn.functional as F



class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs, return_feature=False):
        """
        Args:
            inputs: input tensor
            return_feature: if True, return (logits, h) where h is fc1 output (for FedFed plugin).
        """
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        h = F.relu(self.fc1(tensor))  # intermediate feature (B, 512)
        logits = self.fc2(h)
        if return_feature:
            return logits, h
        return logits
