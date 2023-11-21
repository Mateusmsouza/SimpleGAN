import itertools
import math

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.descriminator import Descriminator
from models.generator import Generator

bath_size = 32
epochs = 100
learning_rate = 2e-3

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def display_samples(image_matrix, num_samples=9, image_width=28, image_length=28):
    size_figure_grid = int(math.sqrt(num_samples))
    fig, ax = plt.subplots(3, 3, figsize=(6, 6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    # reshaped_generated_images = fake_images.view(batch_size, 28, 28)
    
    for k in range(num_samples):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i,j].cla()
        ax[i,j].imshow(image_matrix.detach().numpy()[k,:].reshape(image_width, image_length), cmap='Greys_r')
    plt.show()

def descriminator_loss(images, synthetic_images, descriminator):
    im_out = descriminator(images)
    sim_out = descriminator(synthetic_images)
    im_loss = F.binary_cross_entropy(im_out, torch.ones(im_out.shape))
    sim_loss = F.binary_cross_entropy(sim_out, torch.zeros(sim_out.shape))
    return im_loss + sim_loss

def generator_loss(synthetic_images, descriminator):
    sim_out = descriminator(synthetic_images)
    loss = F.binary_cross_entropy(sim_out, torch.ones(sim_out.shape))
    return loss


def train_loop(epochs: int, training_dataloader, generator, descriminator, optimizer_d, optimizer_g):
    for epoch in range(epochs):
        backprop_g = (epoch % 2) == 0
        for data, _ in training_dataloader:
            z = torch.randn(data.shape)
            synthetic_images = generator(z)

            if backprop_g:
                optimizer_g.zero_grad()
                loss_gn = generator_loss(synthetic_images, descriminator)
                loss_gn.backward() # retain_graph=True
                optimizer_g.step()
            else:
                optimizer_d.zero_grad()
                loss_dc = descriminator_loss(data, synthetic_images, descriminator)
                loss_dc.backward()
                optimizer_d.step()
        if backprop_g:
            display_samples(generator(torch.randn([9, 1, 28, 28])))
        print(f'Epoch {epoch}: Descriminator loss: {loss_dc.item() if not backprop_g else None}, Generator loss: {loss_gn.item() if backprop_g else None}')
        #print(f'Descriminator weights: fc1 {descriminator.fc1.weight} fc2 {descriminator.fc2.weight}')

if __name__ == '__main__':
    dataset = MNIST('./datasets/mnist', train=True, download=True, transform=train_transforms)
    training_dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=True)
    descriminator = Descriminator()
    generator = Generator()
    optimizer_d = Adam(descriminator.parameters(), lr=learning_rate)
    optimizer_g = Adam(generator.parameters(), lr=learning_rate)
    train_loop(epochs=epochs, training_dataloader=training_dataloader, generator=generator, descriminator=descriminator, optimizer_d=optimizer_d, optimizer_g=optimizer_g)