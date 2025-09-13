# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 21:15:35 2025

@author: Sai Gunaranjan
"""

"""
In this script, I have implemented a vanilla GAN on MNIST dataset. GAN stands for Generative Adverserial Networks.
I have used the architecture as mentioned in the tutorials of Deep Generative Models by Pratosh.
Link:
    https://www.youtube.com/watch?v=iOb8vmlJd8o&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=13
The implementation is in pytorch. The code is well documented to explain each and every step.
But the results are not so great because, I have done a simple flattening of the images and feeding into the
Discriminator network which is a Feed Forward neural network. Similarly, at the Generator, I'm using a FFNN and finally reshaping
the ouput to a 28 x 28 image. Ideally, I should be using convolution NNs to process images. I wil be doing this in my next
where I will be implementing a Conditional Deep Convolutional GAN. This implementation is to get started on GAN implementation
and get hands-on pytorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import time as time


# Define transform to perform on dataset
transform = transforms.Compose([transforms.ToTensor(), # Converts Uint8 to float [0,1]
                                transforms.Normalize((0.5,), (0.5,))]) # Subtracts -1/2 and scales by 2 to bring to [-1,1]

# Load MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True,transform=transform,download=True)
# test_dataset = datasets.MNIST(root="./data", train=False,transform=transform,download=True) # Test data not required in vanlla GANs

# Batch and shuffle the data
batchSize = 128
train_loader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True)


# Define where to run the code: CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')# for debug
print(device)

saveImagesFolder = 'vanilla_GAN_generated_images'
os.makedirs(saveImagesFolder, exist_ok=True)

#%%
# Define Generator architecture
class Generator(nn.Module):

    def __init__(self,z_dim,imageDim):

        super(Generator,self).__init__() # child class Generator inherits all attributes of parent class nn.Module
        # Define Generator architecture
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, imageDim),
            nn.Tanh() # to bring back to [-1,1] range
            )

    def forward(self, z): # Forward pass in Generator
        return self.model(z)


# Define Discriminator architecture
class Discriminator(nn.Module):

    def __init__(self,imageDim):

        super(Discriminator,self).__init__() # child class Discriminator inherits all attributes of parent class nn.Module
        # Define Discriminator architecture
        self.model = nn.Sequential(
            nn.Linear(imageDim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Binary classifier 0 or 1
            )

    def forward(self,image):

        return self.model(image)


z_dim = 100
imageDim = 28*28
# Create an object of Generator and Discriminator
gen = Generator(z_dim,imageDim)
disc = Discriminator(imageDim)

# Move Generator model and Discriminator model to GPU
gen = gen.to(device)
disc = disc.to(device)

# Define optimizer to use and the loss function to use for Generator and Discriminator networks
criterion = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(),  lr=2e-4)
optimizer_disc = optim.Adam(disc.parameters(), lr=1e-4)



#%%
numGeneratedImages = 16
epochs = 50
numDiscTrain = 1
numGenTrain = 2

# Start training loop across epochs
for epoch in range(epochs):
    tstart = time.time()
    # Set Discriminator model in train mode
    disc.train()
    # Set Generator model in train mode
    gen.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        internalBatchSize = data.shape[0]
        # Train Discriminator

        # Set the Discriminator gradients to 0 at the beginning of each batch computation
        optimizer_disc.zero_grad()

        actualImages = data.reshape(internalBatchSize,imageDim)
        actualImages = actualImages.to(device)

        # Perform forward pass on Discriminator with real samples
        predClassLabelsTrueImages = disc(actualImages)
        # Set class labels = 1 for real data samples
        trueClassLabelsTrueImages = torch.ones((internalBatchSize),device=device)[:,None]
        # Compute loss for Discriminator on real samples
        lossAtDiscFromTrueImages = criterion(predClassLabelsTrueImages,trueClassLabelsTrueImages)

        # Perform forward pass on Discriminator with fake samples from generator
        z = torch.randn(internalBatchSize,z_dim)
        z = z.to(device) # Move data to device
        # 1. Compute forward pass on generator
        fakeImages = gen(z)
        # 2. Compute forward pass on Discriminator with output of Generator
        # Detach the input while feeding into the Discriminator to disconnect the computational graph and avoid gradient computation for generator
        predClassLabelFakeImages = disc(fakeImages.detach())
        # 3. Set class labels = 0 for fake data coming from generator
        trueClassLabelFakeImages = torch.zeros((internalBatchSize),device=device)[:,None]
        # 4. Compute Loss for Discriminator on fake samples
        lossAtDiscFromFakeImages = criterion(predClassLabelFakeImages,trueClassLabelFakeImages)

        # Sum of Discriminator losses from true images and fake images
        totalDiscriminatorLoss = lossAtDiscFromFakeImages +  lossAtDiscFromTrueImages
        # Backpropagate total discriminator loss on true and fake images through Discriminator only
        totalDiscriminatorLoss.backward()
        # Update weights of Discriminator only
        optimizer_disc.step()


        # Train Generator for k = p times of Discriminator training to avoid
        #Discriminator from hitting 100% training accuracy resulting in saturation

        for ele in range(numGenTrain):
            # Make gradients of Generator 0 at the beginning
            optimizer_gen.zero_grad()

            z = torch.randn(internalBatchSize,z_dim)
            z = z.to(device) # Move data to device

            # 1. compute forward pass on generator
            fakeImages = gen(z)
            # 2. forward pass on discriminator
            predClassLabelFakeImages = disc(fakeImages)
            # 3. Set class labels = 1 for fake data coming from generator. This is because generator is trying to fool discriminator
            # into thinking that the images generated are actual images
            trueClassLabelFakeImages = torch.ones((internalBatchSize),device=device)[:,None] # This is a very important step
            # 4. Compute Loss for generator at Discriminator output
            genLossAtDisc = criterion(predClassLabelFakeImages,trueClassLabelFakeImages) # Store loss as well!
            # 5. Back propagate loss through Discriminator and then through Generator and compute gradients wrt Generator parameters.
            # Pytorch automatically takes care of this
            genLossAtDisc.backward()
            # 6. Update the weights of generator and dont update weights of Discriminator
            optimizer_gen.step()
            # print(f"Generator loss at epoch {epoch} / {epochs} = {genLossAtDisc:4f}")


    # Generate an image every N epochs
    if ((epoch+1)%10 == 0): # ((epoch+1)%10 == 0):
        print(f"Epoch [{epoch+1}/{epochs}], D_loss: {totalDiscriminatorLoss.item():.4f}, G_loss: {genLossAtDisc.item():.4f}")
        # Set Generator model in eval mode
        gen.eval()
        with torch.no_grad():
            z = torch.randn(numGeneratedImages,z_dim)
            z = z.to(device)
            generatedImageFlatten = gen(z)

        generatedImageFlattenTransformed = ((generatedImageFlatten+1)/2) # Bring back to range [0,1]
        # generatedImageFlattenTransformed = generatedImageFlattenTransformed.to(torch.uint8)
        generatedImage = generatedImageFlattenTransformed.reshape(numGeneratedImages,1,28,28)
        generatedImage = generatedImage.cpu()

        save_image(generatedImage, saveImagesFolder + '//' + "generated_image_epoch_{}.png".format(epoch+1),
                   nrow=4, normalize=False) # Automatically converts to uint8

    tend = time.time()

    timeEachEpoch = (tend - tstart)
    # print('Time taken for training epoch {0} / {1} = {2:.1f} sec'.format(epoch+1,epochs,timeEachEpoch))


















