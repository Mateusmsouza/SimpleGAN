import itertools
import math

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from models.discriminator import Discriminator
from models.generator import Generator

bath_size = 32
epochs = 50
learning_rate = 3e-4
z_dim = 64
image_dim = 784 # 28 * 28

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# def display_samples(image_matrix, num_samples=9, image_width=28, image_length=28):
#     size_figure_grid = int(math.sqrt(num_samples))
#     fig, ax = plt.subplots(3, 3, figsize=(6, 6))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i,j].get_xaxis().set_visible(False)
#         ax[i,j].get_yaxis().set_visible(False)
#     # reshaped_generated_images = fake_images.view(batch_size, 28, 28)
    
#     for k in range(num_samples):
#         i = k // size_figure_grid
#         j = k % size_figure_grid
#         ax[i,j].cla()
#         ax[i,j].imshow(image_matrix.detach().numpy()[k,:].reshape(image_width, image_length), cmap='Greys_r')
#     plt.show()

criterion = nn.BCELoss()
writer = SummaryWriter()
fixed_noise = torch.randn(1, z_dim)
def train_loop(epochs: int, training_dataloader, generator, discriminator, optimizer_d, optimizer_g):
    for epoch in range(epochs):
        for data, _ in training_dataloader:
            data = data.view(-1, 784)
            #print(data.shape[0])
            z = torch.randn(data.shape[0], z_dim)
            synthetic_images = generator(z)
            discriminator_output_real = discriminator(data).view(-1)
            loss_d = criterion(discriminator_output_real, torch.ones_like(discriminator_output_real))
            discriminator_output_synthetic = discriminator(synthetic_images).view(-1)
            loss_d_synthetic = criterion(discriminator_output_synthetic, torch.zeros_like(
                discriminator_output_synthetic))
            loss_d = (loss_d + loss_d_synthetic) / 2

            discriminator.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_d.step()

            output = discriminator(synthetic_images).view(-1)
            loss_g = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss_g.backward()
            optimizer_g.step()
        print(f"logging epoch {epoch}")
        with torch.no_grad():
            writer.add_scalar("Loss/train-discriminator", loss_d.item(), epoch)
            writer.add_scalar("Loss/train-generator", loss_g.item(), epoch)
            writer.add_image("Generated image", generator(fixed_noise).reshape(1, 28, 28), epoch)

if __name__ == '__main__':
    dataset = MNIST('./datasets/mnist', train=True, download=True, transform=train_transforms)
    training_dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=True)
    discriminator = Discriminator(image_dim)
    generator = Generator(z_dim, image_dim)
    optimizer_d = Adam(discriminator.parameters(), lr=learning_rate)
    optimizer_g = Adam(generator.parameters(), lr=learning_rate)
    train_loop(epochs=epochs, training_dataloader=training_dataloader, generator=generator, discriminator=discriminator, optimizer_d=optimizer_d, optimizer_g=optimizer_g)