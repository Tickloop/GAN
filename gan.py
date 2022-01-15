import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm # this is used for progress bars
import numpy as np
import torch.nn as nn
import torch

img_size = 28
latent_dim = 64
batch_size = 100
MAX_EPOCH = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Generator(nn.Module):
    """
        The Generator class is responsible for generating new data which must pass through the discriminator.
        To generate new data, random noise input is required.
        The output size in 512 x 512 or N x 512 x 512 for batch processing.
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        img = self.model(x)
        img = torch.reshape(img, (batch_size, img_size, img_size))
        return img

class Discriminator(nn.Module):
    """
        The Discrimniator class is responsible for binary classification.
        Final output size is 1 or N x 1 for batch processing.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        is_valid = self.model(img)
        return is_valid

def dataLoader(images):
    """
        Function is responsible for reading image files from data/mnist/trainingSet/trainingSet/0 directory
        and resizing them to 512 x 512 so they can be fed into our GAN for training.
    """
    for path, dirs, filenames in os.walk('data/mnist/trainingSet/trainingSet/0'):
        for filename in tqdm(filenames):
            img = cv2.imread(f"{path}/{filename}", 0)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)

def train(generator, discriminator, images):
    # optimizers that we will use
    generater_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # the loss function
    adv_loss = nn.BCELoss()

    # total number of real data points that we have
    N = images.shape[0]

    for epoch in range(MAX_EPOCH):
        z_mini_batch = torch.Tensor(np.random.normal(0, 1, size=(batch_size, latent_dim))) # size (batch_size x laten_dim)
        z_mini_batch = z_mini_batch.to(device)
        
        random_choice = np.random.choice(np.arange(N), size=batch_size) # size (batch_size x 1)
        real_images = torch.Tensor(np.array([images[choice] for choice in random_choice])) # size (batch_size x img_size x img_size x channels)
        real_images = real_images.to(device)
              

        # labels for real and fake data
        real_data_labels = torch.Tensor(np.ones(shape=(batch_size, 1))) # size (batch_size x 1)
        real_data_labels = real_data_labels.to(device)
        fake_data_labels = torch.Tensor(np.zeros(shape=(batch_size, 1))) # size (batch_size x 1)
        fake_data_labels = fake_data_labels.to(device)

        # train the discriminator
        discriminator_optimizer.zero_grad()
        fake_images = generator(z_mini_batch) # size (batch_size x img_size x img_size x channels)
        real_loss = adv_loss(discriminator(real_images), real_data_labels)
        fake_loss = adv_loss(discriminator(fake_images), fake_data_labels)
        discriminator_loss = (real_loss + fake_loss) / 2

        # backprop
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # train the generator
        generator.zero_grad()

        z_mini_batch = torch.Tensor(np.random.normal(0, 1, size=(batch_size, latent_dim))) # size (batch_size x laten_dim)
        z_mini_batch = z_mini_batch.to(device)

        fake_images = generator(z_mini_batch) # size (batch_size x img_size x img_size x channels)

        """ we pass real data labels here since we want the discriminator to identify these images as real """
        generator_loss = adv_loss(discriminator(fake_images), real_data_labels)
        
        # backprop
        generator_loss.backward()
        generater_optimizer.step()

        print(f"Epoch: {epoch + 1} / {MAX_EPOCH}, Generator Loss: {generator_loss} , Discriminator Loss: {discriminator_loss}")

        if epoch % 5 == 0:
            cv2.imwrite(f"./samples/0/sample_epoch_{epoch}.jpg", fake_images[0].cpu().detach().numpy() * 255)

def main():
    images = []

    # load the data from images 
    dataLoader(images)
    images = np.array(images)
    print("Images loaded", images.shape)
        
    generator = Generator()
    discriminator = Discriminator()

    # let's go faster
    generator.to(device)
    discriminator.to(device)

    train(generator, discriminator, images)

if __name__ == "__main__":
    main()