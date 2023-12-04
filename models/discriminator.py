import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, image_dimension):
        super(Discriminator, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(image_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''x is image'''
        return self.fc_seq(x)
