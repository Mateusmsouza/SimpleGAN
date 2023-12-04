import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, z_dim, image_dimension):
        super(Generator, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, image_dimension),
            nn.Tanh()
        )

    def forward(self, z):
        '''z is noise'''
        return self.fc_seq(z)
