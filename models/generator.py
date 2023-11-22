import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, image_dimension):
        super(Generator, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(image_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, image_dimension)
        )

    def forward(self, z):
        '''z is noise'''
        return self.fc_seq(z)
