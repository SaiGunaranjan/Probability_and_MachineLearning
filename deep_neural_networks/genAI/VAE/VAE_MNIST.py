# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 12:06:34 2025

@author: Sai Gunaranjan
"""


# q_phi (z|x) = N(z, mu_phi, Sigma_phi)
# p(z) = N(0,I)
# P_theta (x|z) = soft class labels

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
import time as time



# Define the tranforms on the data as part of preprocessing
transform = transforms.Compose([transforms.ToTensor()]) # Bringgs data from [0,255] to [0,1]

# Batch and load the data
train_dataset = datasets.MNIST(root="./data", train = True,transform=transform, download = False)

# Batch the data
batchSize = 128#128
train_loader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True)

# Define device to run the code on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saveImagesFolder = 'VAE_generated_images'
os.makedirs(saveImagesFolder, exist_ok=True)

# Define Encoder architecture
class Encoder(nn.Module):

    def __init__(self,latent_space_dim):

        super(Encoder,self).__init__() # Make child class Encoder inherit the properties of parent class nn.Module

        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), # output size = 14 x 14
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1), # Output size = 7 x 7
            nn.ReLU(True)
            )
        # Flattening required at this stage to feed into FC layers
        self.fc_model = nn.Sequential(
            nn.Linear(7*7*64, 256),
            nn.ReLU(True)
            )

        # Obtain separate outputs for mu and sigma
        self.mu_phi = nn.Linear(256, latent_space_dim)
        self.log_variance_phi = nn.Linear(256, latent_space_dim) # Assume Covariance is a diagonal matrix of individual variances.
        # Hence only latent_space_dim number of varicne parameters need to be computed.
        # Also, we compute log variances and not just variances because, computing varices means forcing this output to be strictly non-negative.
        # A more relaxed approch is to compute log variances since this does not have any contraints. It can be positive or negative.
        # And then get back the variances by taking antilog.


    def forward(self, image_data):

        conv_layer_output = self.conv_model(image_data)
        conv_layer_output_flatten = conv_layer_output.flatten(start_dim=1) # Flatten the output of the conv layer
        fc_layer_output = self.fc_model(conv_layer_output_flatten)

        mu_phi = self.mu_phi(fc_layer_output)
        log_variance_phi = self.log_variance_phi(fc_layer_output)


        return mu_phi, log_variance_phi


# Define architecture for Decoder
class Decoder(nn.Module):

    def __init__(self, latent_space_dim):

        super(Decoder,self).__init__() # Make child class Encoder inherit the properties of parent class nn.Module

        self.fc_model = nn.Sequential(
            nn.Linear(latent_space_dim, 256),
            nn.ReLU(True),

            nn.Linear(256, 7*7*64),
            nn.ReLU(True)
            )

        # Convert 1d data to image data i.e 2d

        self.conv_model = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # Output size = 14 x 14
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1), # Output size = 28 x 28
            nn.ReLU(True),

            nn.Conv2d(32, 1, 3, 1, 1), # Output size = 28 x 28
            nn.Sigmoid() # Outputs probabilities for each of the pixels
            )


    def forward(self, latent_variable_z):

        fc_layer_output = self.fc_model(latent_variable_z)
        fc_layer_output = fc_layer_output.reshape(-1,64,7,7) # Reshape as batch x 64 channels x 7 x 7

        # Output of the decoder are the parameters of the distribution P_\theta (x|z). But in this case,
        # the parameters are the probabilities and the reconstructed image as well
        reconst_image = self.conv_model(fc_layer_output) # With each pixel as a probability value [0,1]

        return reconst_image


latent_space_dim = 20
num_instances_epsillon = 1 # Number of instances of epsillon for each data point X. This is used to get the Expectation of log likelihood P_\theta(x|z)
beta = 1 # To model beta VAE

# Define the Encoder and Decoder objects
encoder = Encoder(latent_space_dim)
decoder = Decoder(latent_space_dim)

# Move model to device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Define Loss Function as BCE for the log likelihood or reconstruction term
criterion = nn.BCELoss(reduction='sum')

optimizer_encoder = optim.Adam(encoder.parameters(), lr=1e-3)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=1e-3)


epochs = 30#100
numGeneratedImages = 16

if __name__ == "__main__":
    # Start training loop

    for epoch in range(epochs):
        tstart = time.time()
        # Set Encoder and Decoder in train mode
        encoder.train()
        decoder.train()

        for batch_idx, (image_data, _) in enumerate(train_loader):
            # print('batch_idx', batch_idx)
            # Set gradients to 0
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            internalBatchSize = image_data.shape[0]

            # Move data to device
            image_data = image_data.to(device)

            # Train Encoder

            # Compute forward pass on Encoder
            mu_phi, log_variance_phi = encoder(image_data)

            # May not require gradients for the below steps
            with torch.no_grad():
                epsillon = torch.randn(internalBatchSize,num_instances_epsillon, latent_space_dim) # Draw samples from N(0,I)
                epsillon = epsillon.to(device) # Move epsillon also to device to perform operations with other tensors on device

            variance_phi = torch.exp(log_variance_phi) # Get back variances from log variances
            sigma_phi = torch.sqrt(variance_phi)

            # Reparameterization trick
            latent_variable_z = mu_phi[:,None,:] + (sigma_phi[:,None,:] * epsillon) # This samples z from q_phi (z|x) indirectly through the reparameterization trick

            # Flatten tensor to batchsize*num_instances_epsillon x z_dim to feed to Decoder
            latent_variable_z = latent_variable_z.reshape(-1,latent_space_dim)

            # Compute forward pass on Decoder
            reconst_image = decoder(latent_variable_z) # internalBatchSize*num_instances_epsillon, 1, 28, 28

            reconst_image = reconst_image.reshape(internalBatchSize,num_instances_epsillon, 1, 28, 28)

            # Make multiple copies(num_instances_epsillon number of copies) of the input image for each example data in the batch
            image_data_expand = image_data.unsqueeze(1).expand(-1, num_instances_epsillon, -1, -1, -1)

            # Compute BCE loss for the reconstruction/log likelihood term
            loss_recon_term = criterion(reconst_image,image_data_expand)/(internalBatchSize*num_instances_epsillon)

            # Compute KL term
            kl_divergence = 1/2 * (torch.sum(variance_phi,axis=1) + torch.sum(mu_phi**2,axis=1) - latent_space_dim -
                                   torch.sum(log_variance_phi,axis=1))
            kl_divergence = torch.mean(kl_divergence) # Compute mean across batch dimension

            # Total Encoder loss = loss from BCE (or negative of log likelihood) + beta*KL(q_phi(z|x) || P(z))
            total_loss_encoder = loss_recon_term + beta*kl_divergence
            # Perform the backward pass from the loss
            total_loss_encoder.backward()
            # Update parameters of Encoder keeping weights of Decoder constant
            optimizer_encoder.step()

            ################################################################################
            # Train Decoder

            # Compute forward pass on Decoder
            mu_phi, log_variance_phi = encoder(image_data)

            # May not require gradients for the below steps
            with torch.no_grad():
                epsillon = torch.randn(internalBatchSize,num_instances_epsillon, latent_space_dim) # Draw samples from N(0,I)
                epsillon = epsillon.to(device) # Move epsillon also to device to perform operations with other tensors on device

            variance_phi = torch.exp(log_variance_phi) # Get back variances from log variances
            sigma_phi = torch.sqrt(variance_phi)

            # Reparameterization trick
            latent_variable_z = mu_phi[:,None,:] + (sigma_phi[:,None,:] * epsillon) # This samples z from q_phi (z|x) indirectly through the reparameterization trick

            # Flatten tensor to batchsize*num_instances_epsillon x z_dim to feed to Decoder
            latent_variable_z = latent_variable_z.reshape(-1,latent_space_dim)

            # Compute forward pass on Decoder
            # Detach the input while feeding into the Decoder to disconnect the computational graph and avoid
            #gradient computation for the Encoder
            reconst_image = decoder(latent_variable_z.detach()) # internalBatchSize*num_instances_epsillon, 1, 28, 28

            reconst_image = reconst_image.reshape(internalBatchSize,num_instances_epsillon, 1, 28, 28)

            # Make multiple copies(num_instances_epsillon number of copies) of the input image for each example data in the batch
            image_data_expand = image_data.unsqueeze(1).expand(-1, num_instances_epsillon, -1, -1, -1)

            # Compute BCE loss for the reconstruction/log likelihood term
            loss_recon_term = criterion(reconst_image,image_data_expand)/(internalBatchSize*num_instances_epsillon)

            # KL term does not contribute to the Decoder loss since it is not a function of theta

            # Total Decoder loss = loss from BCE (or negative of log likelihood)
            total_loss_decoder = loss_recon_term
            # Perform the backward pass from the loss
            total_loss_decoder.backward()
            # Update parameters of Encoder keeping weights of Decoder constant
            optimizer_decoder.step()



        # Generate an image every N epochs
        if ((epoch+1)%2 == 0):
            print(f"Epoch [{epoch+1}/{epochs}], Encoder_loss: {total_loss_encoder.item():.4f}, Decoder_loss: {total_loss_decoder.item():.4f}")
            # Set Decoder models in eval mode
            decoder.eval()
            with torch.no_grad():
                latent_variable_z = torch.randn(numGeneratedImages,latent_space_dim) # Draw samples from a standar normal
                latent_variable_z = latent_variable_z.to(device)
                generatedImageFlatten = decoder(latent_variable_z)


            generatedImage = generatedImageFlatten.reshape(numGeneratedImages,1,28,28)
            generatedImage = generatedImage.cpu()

            save_image(generatedImage, saveImagesFolder + '//' + "generated_image_epoch_{}.png".format(epoch+1),
                       nrow=4, normalize=False) # Automatically converts to uint8



        tend = time.time()

        timeEachEpoch = (tend - tstart)
        print('Time taken for training epoch {0} / {1} = {2:.1f} sec'.format(epoch+1,epochs,timeEachEpoch))










