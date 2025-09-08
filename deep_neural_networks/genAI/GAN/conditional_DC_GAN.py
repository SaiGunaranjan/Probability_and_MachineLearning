# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:57:40 2025

@author: Sai Gunaranjan
"""


"""
In this script, I have implemented a conditional DC GAN on MNIST dataset.  Conditional DC GAN stands for
Conditional Deep Convolutional Generative Adverserial Networks. The conditioning is based on the class labels [0,9].
I have used the architecture proposed in the original paper on conditonal DC GANs by (Mirza & Osindero, 2014).
Im able to generate correct images of whatever digit I feed in. The implementation is in pytorch. The code is well documented
to explain each and every step
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
import time as time
import matplotlib.pyplot as plt


#%%
# Helper function for plotting

def save_images_with_labels(images, labels, filename="mnist_grid.png", nrow=4, figsize=(8,8), dpi=200):
    """
    Save a grid of images with labels.

    Args:
        images (Tensor): Shape [N, 1, H, W] or [N, H, W], values in [0,1] or [0,255].
        labels (Tensor or list): Length N, class labels.
        filename (str): Output filename (e.g., "grid.png").
        nrow (int): Number of images per row (default=4).
        figsize (tuple): Figure size in inches.
        dpi (int): Resolution of saved image.
    """
    images = images.detach().cpu()
    labels = [str(l) for l in labels]

    ncol = nrow
    n_images = len(images)
    nrow = (n_images + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_images:
            img = images[i].squeeze()   # remove channel dimension if [1,H,W]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Generate: {labels[i]}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()

#%%
# Define transform to perform on dataset
transform = transforms.Compose([transforms.ToTensor(), # To bring it from uint8 to [0,1]]
                                transforms.Normalize((0.5,),(0.5,))]) # to bring it from [0,1] to [-1,1]

# Load the dataset
train_dataset = datasets.MNIST(root="./data", train=True,transform=transform,download=True)

# Batch the data
batchSize = 128
train_loader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True)

# Define where to run the code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saveImagesFolder = 'conditional_DC_GAN_generated_images'
os.makedirs(saveImagesFolder, exist_ok=True)

#%%
# Define the Generator architecture
class Generator(nn.Module):

    def __init__(self,z_dim, num_class_labels, emb_dim):

        super(Generator,self).__init__() # Make the child class Generator inherit the attributes of parent nn.Module class
        # Define all the parameters and architecture in the init method

        self.embedding = nn.Embedding(num_embeddings = num_class_labels, embedding_dim = emb_dim)

        self.fc = nn.Linear(z_dim + emb_dim, 7*7*256)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), # No bias. output size = 14 x 14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,4,2,1), # No bias. output size = 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,1,3,1,1), # output size = 28 x 28 x 1
            nn.Tanh()
            )

    def forward(self,z, class_labels):
        # num of rows od z and number of class labels should be same! We need to pass in the class labels for each row of z

        embed_vec = self.embedding(class_labels) # Get the embeddings for each of the class labels for Generator
        x = torch.cat((z,embed_vec),dim=1) # Concatenate z and embeddings
        x = self.fc(x)
        x = x.reshape(-1,256,7,7) # Convert the flattened FC layer output to a image

        return self.model(x)

# Define the Discriminator architecture

class Discriminator(nn.Module):

    def __init__(self, num_class_labels, embed_dim):

        super(Discriminator,self).__init__() # Make the child class Discriminator inherit the attributes of parent nn.Module class

        self.embedding = nn.Embedding(num_embeddings=num_class_labels, embedding_dim = embed_dim)
        self.fc = nn.Linear(embed_dim,28*28) # Hardcoding the maping to 28 x 28 = 784 for MNIST

        self.conv_layer = nn.Sequential(
            nn.Conv2d(2,64,4,2,1), # output size = 14 x 14
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,4,2,1), # output size = 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,3,2,1), # output size = 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            )

        self.linear_layer = nn.Sequential(
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
            )

    def forward(self,image, class_labels):

        embed_vec = self.embedding(class_labels) # Get the embeddings for each of the class labels for Generator
        x = self.fc(embed_vec) # Convert embed_dim to 28*28 through linear layer
        x = x.reshape(-1,28,28)
        x = x[:,None,:,:] # Batchsize x 1 channel x 28 x 28
        class_aug_image = torch.cat((image,x),dim=1) # Augment the mapped class embedding as a channel to the input image
        conv_layer_output = self.conv_layer(class_aug_image)
        conv_layer_output = conv_layer_output.flatten(start_dim=1) # Flatten the output of the conv layer
        disc_pred_output = self.linear_layer(conv_layer_output) # Pass the flattened conv layer output to the final linear layer with sigmoid activation

        return disc_pred_output

#%%

z_dim = 100
num_class_labels = 10 # For MNIST dataset
emb_dim = 50 # For both Generator and Discriminator

gen = Generator(z_dim,num_class_labels,emb_dim) # Define Generator object
disc = Discriminator(num_class_labels,emb_dim) # Define Generator object

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move Generator and Discriminator models to the device
gen = gen.to(device)
disc = disc.to(device)

# Define the loss function
criterion = nn.BCELoss()

# Define optimizers for Generator and Discriminator parameters
beta1 = 0.5
beta2 = 0.999
optimizer_gen = optim.Adam(gen.parameters(), lr= 2e-4, betas=(beta1,beta2))
optimizer_disc = optim.Adam(disc.parameters(), lr= 2e-4, betas=(beta1,beta2))


#%%
numGeneratedImages = 16
epochs = 100
numDiscTrain = 1
numGenTrain = 1

if __name__ == "__main__":
    # Start training loop across epochs
    for epoch in range(epochs):
        tstart = time.time()
        # Set Discriminator model in train mode
        disc.train()
        # Set Generator model in train mode
        gen.train()
        for batch_idx, (actualImages, class_labels) in enumerate(train_loader):
            internalBatchSize = actualImages.shape[0]
            # Train Discriminator

            #Set the Discriminator gradients to 0 at the beginning of each batch computation
            optimizer_disc.zero_grad()

            # Move data to device
            actualImages = actualImages.to(device)
            class_labels = class_labels.to(device)

            #Perform forward pass on Discriminator with real samples and corresponding class labels
            predClassLabelsTrueImages = disc(actualImages,class_labels)
            # Set class labels = 1 for real data samples
            trueClassLabelsTrueImages = torch.ones((internalBatchSize),device=device)[:,None]
            # Compute loss for Discriminator on real samples
            lossAtDiscFromTrueImages = criterion(predClassLabelsTrueImages,trueClassLabelsTrueImages)

            # Perform forward pass on Discriminator with fake samples from generator
            z = torch.randn(internalBatchSize,z_dim)
            z = z.to(device) # Move data to device

            # Define random class labels for the fake images to be generated by Generator
            class_labels_gen = torch.randint(num_class_labels, (internalBatchSize,))
            class_labels_gen = class_labels_gen.to(device)
            # 1. Compute forward pass on generator
            fakeImages = gen(z, class_labels_gen)
            # 2. Compute forward pass on Discriminator with output of Generator
            # Detach the input while feeding into the Discriminator to disconnect the computational graph and avoid gradient computation for generator
            predClassLabelFakeImages = disc(fakeImages.detach(), class_labels_gen)
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


            # Train Generator for k = p times of Discriminator training to avoid Discriminator from hitting 100% training accuracy resulting in saturation

            for ele in range(numGenTrain):
                # Make gradients of Generator 0 at the beginning
                optimizer_gen.zero_grad()

                z = torch.randn(internalBatchSize,z_dim)
                z = z.to(device) # Move data to device

                # 1. compute forward pass on generator
                fakeImages = gen(z, class_labels_gen)
                # 2. forward pass on discriminator
                predClassLabelFakeImages = disc(fakeImages, class_labels_gen)
                # 3. Set class labels = 1 for fake data coming from generator. This is because generator is trying to fool discriminator
                #into thinking that the images generated are actual images
                trueClassLabelFakeImages = torch.ones((internalBatchSize),device=device)[:,None] # This is a very important step
                # 4. Compute Loss for generator at Discriminator output
                genLossAtDisc = criterion(predClassLabelFakeImages,trueClassLabelFakeImages) # Store loss as well!
                # 5. Back propagate loss through Discriminator and then through Generator and compute gradients wrt Generator parameters.
                #Pytorch automatically takes care of this
                genLossAtDisc.backward()
                # 6. Update the weights of generator and dont update weights of Discriminator
                optimizer_gen.step()
                # print(f"Generator loss at epoch {epoch} / {epochs} = {genLossAtDisc:4f}")


        # Generate an image every N epochs
        if ((epoch+1)%5 == 0): # ((epoch+1)%10 == 0):
            print(f"Epoch [{epoch+1}/{epochs}], D_loss: {totalDiscriminatorLoss.item():.4f}, G_loss: {genLossAtDisc.item():.4f}")
            # Set Generator model in eval mode
            gen.eval()
            with torch.no_grad():
                z = torch.randn(numGeneratedImages,z_dim)
                z = z.to(device)
                class_labels_gen = torch.randint(num_class_labels, (numGeneratedImages,))
                class_labels_gen = class_labels_gen.to(device)
                generatedImage = gen(z, class_labels_gen)

            generatedImageTransformed = ((generatedImage+1)/2) # Bring back to range [0,1]
            generatedImage = generatedImageTransformed.cpu()

            class_labels_str = [f' {i}' for i in class_labels_gen]

            save_images_with_labels(generatedImage, class_labels_str,
                                    saveImagesFolder + '//' + "generated_image_epoch_{}.png".format(epoch+1),
                                    nrow=4, figsize=(8,8), dpi=400)

        tend = time.time()

        timeEachEpoch = (tend - tstart)
        # print('Time taken for training epoch {0} / {1} = {2:.1f} sec'.format(epoch+1,epochs,timeEachEpoch))



    # Post training, save the model weights for inference later on
    torch.save(gen.state_dict(),"generator.pth") # For generation/Inference, only Generator is required!










