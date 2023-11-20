import torch
import torch.nn as nn
import torch.nn.functional as F


class Descriminator(nn.Module):

    def __init__(self, image_size: int = 28):
        super(Descriminator, self).__init__()
        self.image_size = image_size
        
        self.fc1 = nn.Linear(image_size, image_size)
        self.fc2 = nn.Linear(image_size, image_size)

    def forward(self, x):
        '''x is image'''
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.sigmoid(output)
        return output