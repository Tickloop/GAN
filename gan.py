import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm # this is used for progress bars
import numpy as np
import torch.nn as nn
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to run the network for")
parser.add_argument("--latent_dim", default=64, type=int, help="The size of the noise vector to pass in the Generator")
parser.add_argument("--batch_size", default=100, type=int, help="Size of mini batch that is fed into one step of network training")
parser.add_argument("--img_size", default=28, type=int, help="Size of generated images")
parser.add_argument("--digit", default=0, type=int, help="MNIST digit to be generated")
options = parser.parse_args()
print("Selected arguments to run: ")
print(options)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Generator(nn.Module):
    """
        The Generator class is responsible for generating new data which must pass through the discriminator.
        To generate new data, random noise input is required.
        The output size in img_size x img_size or N x img_size x img_size for batch processing.
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(options.latent_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, options.img_size * options.img_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        img = self.model(x)
        img = torch.reshape(img, (options.batch_size, options.img_size, options.img_size))
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
            nn.Linear(options.img_size * options.img_size, 512),
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
    """ Function is responsible for reading image files from data/mnist/trainingSet/trainingSet/{mnist_number} directory """
    for path, dirs, filenames in os.walk(f"data/mnist/trainingSet/trainingSet/{options.digit}"):
        for filename in tqdm(filenames):
            img = cv2.imread(f"{path}/{filename}", 0)
            images.append(img)

def train(generator, discriminator, images):
    # optimizers that we will use
    generater_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # the loss function
    adv_loss = nn.BCELoss()

    # total number of real data points that we have
    N = images.shape[0]

    for epoch in range(options.epochs):
        z_mini_batch = torch.Tensor(np.random.normal(0, 1, size=(options.batch_size, options.latent_dim))) # size (batch_size x laten_dim)
        z_mini_batch = z_mini_batch.to(device)
        
        random_choice = np.random.choice(np.arange(N), size=options.batch_size) # size (batch_size x 1)
        real_images = torch.Tensor(np.array([images[choice] for choice in random_choice])) # size (batch_size x img_size x img_size x channels)
        real_images = real_images.to(device)
              

        # labels for real and fake data
        real_data_labels = torch.Tensor(np.ones(shape=(options.batch_size, 1))) # size (batch_size x 1)
        real_data_labels = real_data_labels.to(device)
        fake_data_labels = torch.Tensor(np.zeros(shape=(options.batch_size, 1))) # size (batch_size x 1)
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

        z_mini_batch = torch.Tensor(np.random.normal(0, 1, size=(options.batch_size, options.latent_dim))) # size (batch_size x laten_dim)
        z_mini_batch = z_mini_batch.to(device)

        fake_images = generator(z_mini_batch) # size (batch_size x img_size x img_size x channels)

        """ we pass real data labels here since we want the discriminator to identify these images as real """
        generator_loss = adv_loss(discriminator(fake_images), real_data_labels)
        
        # backprop
        generator_loss.backward()
        generater_optimizer.step()

        print(f"Epoch: {epoch + 1} / {options.epochs}, Generator Loss: {generator_loss} , Discriminator Loss: {discriminator_loss}")

        cv2.imwrite(f"./samples/{options.digit}/sample_epoch_{epoch}.jpg", fake_images[0].cpu().detach().numpy() * 255)
        

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

    # ensure that there is a place to store generated samples
    if not os.path.isdir("samples"):
        os.mkdir("samples")
    
    if not os.path.isdir(f"samples/{options.digit}"):
        os.mkdir(f"samples/{options.digit}")

    train(generator, discriminator, images)

if __name__ == "__main__":
    main()