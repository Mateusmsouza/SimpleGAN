import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, image_size: int = 28):
        super(Generator, self).__init__()
        self.image_size = image_size
        
        self.fc1 = nn.Linear(image_size, image_size)
        self.fc2 = nn.Linear(image_size, image_size)

    def forward(self, z):
        '''z is noise'''
        x = self.fc1(z)
        x = F.relu(x)
        x = self.fc2(x)
        return x