# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:51:05 2025

@author: Sai Gunaranjan
"""

"""
In this script, I have implemented the VAE on human faces. The specific architecture was suggested to me by chatGPT
and it is based on architectures presented in several influential works and implementation.
The results are surprisingly good! It is able to generate decent human faces! With this, I have successfully
implemented both VAEs and GANs. Next, I will move to DDPMs which are supposedly state of the art.
"""


import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from celeba_dataset import CelebADataset
import time as time
from torchvision.utils import save_image
import torch.optim as optim



# Define transformation to be applied on the RGB images of faces
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])


dataset = CelebADataset(
    img_dir="./data_faces/img_align_celeba",
    transform=transform
)


batch_size = 128
# Define the dataloader to batch the data
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)


# Define where to run the code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saveImagesFolder = 'VAE_generated_faces'
os.makedirs(saveImagesFolder, exist_ok=True)


# Define Encoder
class Encoder(nn.Module):

    def __init__(self, latent_space_dim):

        super(Encoder,self).__init__()

        self.model_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), # batchsize x 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,64,4,2,1), # batchsize x 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,128,4,2,1), # batchsize x 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,256,4,2,1), # batchsize x 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(True)
            )

        self.model_fc = nn.Sequential(
            nn.Linear(4*4*256, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(True)
            )

        self.mu_phi = nn.Linear(512, latent_space_dim)
        self.log_variance_phi = nn.Linear(512,latent_space_dim)


    def forward(self,image_data):

        convlayer_output = self.model_conv(image_data)
        convlayer_output_flatten = convlayer_output.flatten(start_dim=1)
        fclayer_output = self.model_fc(convlayer_output_flatten)

        mu_phi = self.mu_phi(fclayer_output)
        log_variance_phi = self.log_variance_phi(fclayer_output)

        return mu_phi, log_variance_phi


# Define Decoder
class Decoder(nn.Module):

    def __init__(self, latent_space_dim):

        super(Decoder,self).__init__()

        self.model_fc = nn.Sequential(
            nn.Linear(latent_space_dim,512),
            nn.ReLU(True),

            nn.Linear(512,1024),
            nn.ReLU(True),

            nn.Linear(1024, 4*4*256),
            nn.ReLU(True)
            )
        # Reshape 4*4*256 to 256x4x4
        self.model_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # batchsize x 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1), # batchsize x 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1), # batchsize x 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1), # batchsize x 3 x 64 x 64
            nn.Sigmoid() # Output value pixels will be between [0,1]
            )

    def forward(self, latent_variable_z):

        fc_layer_output = self.model_fc(latent_variable_z)
        fc_layer_output = fc_layer_output.reshape(-1,256,4,4) # Reshape as batch x 64 channels x 7 x 7
        # Output of the decoder are the parameters of the distribution P_\theta (x|z).
        reconst_image = self.model_conv(fc_layer_output) # With each pixel as a probability value [0,1]

        return reconst_image


latent_space_dim = 128
num_instances_epsillon = 10 #30 # Number of instances of epsillon for each data point X. This is used to get the Expectation of log likelihood P_\theta(x|z)
beta = 1 # To model beta VAE

encoder = Encoder(latent_space_dim)
decoder = Decoder(latent_space_dim)

encoder = encoder.to(device)
decoder = decoder.to(device)

# Define Loss Function as BCE for the log likelihood or reconstruction term
criterion = nn.BCELoss(reduction='sum') # P_theta (x|z) = soft class labels
# criterion = nn.MSELoss(reduction='sum') # # P_theta (x|z) = N(x;mu_theta(z),I)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

epochs = 30#100
numGeneratedImages = 16

if __name__ == "__main__":

    for epoch in range(epochs):
        tstart = time.time()

        # Set Encoder and Decoder in train mode
        encoder.train()
        decoder.train()

        for batch_idx, image_data in enumerate(data_loader):

            internalBatchSize = image_data.shape[0]

            # Set gradients to 0
            optimizer.zero_grad()
            # Move data to device
            image_data = image_data.to(device)

            # Compute forward pass on encoder
            mu_phi, log_variance_phi = encoder(image_data)

            with torch.no_grad():
                epsillon = torch.randn(internalBatchSize,num_instances_epsillon,latent_space_dim) # Sample from standard normal # P(z)
                epsillon = epsillon.to(device) # Move epsillon also to device to perform operations with other tensors on device

            variance_phi = torch.exp(log_variance_phi) # Get back variances from log variances
            sigma_phi = torch.sqrt(variance_phi)

            # Perform reparameterization trick
            latent_variable_z = mu_phi[:,None,:] + (sigma_phi[:,None,:] * epsillon) # This samples z from q_phi (z|x) indirectly through the reparameterization trick

            # Flatten tensor to batchsize*num_instances_epsillon x z_dim to feed to Decoder
            latent_variable_z = latent_variable_z.reshape(-1,latent_space_dim)

            # Compute forward pass on decoder
            reconst_image = decoder(latent_variable_z) # internalBatchSize*num_instances_epsillon, 1, 28, 28

            reconst_image = reconst_image.reshape(internalBatchSize,num_instances_epsillon, 3, 64, 64)

            # Make multiple copies(num_instances_epsillon number of copies) of the input image for each example data in the batch
            image_data_expand = image_data.unsqueeze(1).expand(-1, num_instances_epsillon, -1, -1, -1)

            # Compute loss for the reconstruction/log likelihood term. Mean should be taken only across batches and num_instances_epsilon. Not across the vector/matrix dimension!
            loss_recon_term = criterion(reconst_image,image_data_expand)/(internalBatchSize*num_instances_epsillon)

            # Compute KL term. This is a closed form expression that can be derived!
            kl_divergence = 1/2 * (torch.sum(variance_phi,axis=1) + torch.sum(mu_phi**2,axis=1) - latent_space_dim -
                                   torch.sum(log_variance_phi,axis=1))
            kl_divergence = torch.mean(kl_divergence) # Compute mean across batch dimension

            # Total loss from Encoder + Decoder = loss from BCE (or negative of log likelihood) + beta*KL(q_phi(z|x) || P(z))
            total_loss = loss_recon_term + beta*kl_divergence
            # Perform the backward pass from the loss
            total_loss.backward()
            # Update parameters of both Encoder and decoder
            optimizer.step()

        #  Generate an image every N epochs
        if ((epoch+1)%1 == 0):
            print(f"Epoch [{epoch+1}/{epochs}], loss: {total_loss.item():.4f}")
            # Set Decoder models in eval mode
            decoder.eval()
            with torch.no_grad():
                latent_variable_z = torch.randn(numGeneratedImages,latent_space_dim) # Draw samples from a standard normal
                latent_variable_z = latent_variable_z.to(device)
                generatedImageFlatten = decoder(latent_variable_z)


            generatedImage = generatedImageFlatten.reshape(numGeneratedImages,3,64,64)
            generatedImage = generatedImage.cpu()

            save_image(generatedImage, saveImagesFolder + '//' + "generated_image_epoch_{}.png".format(epoch+1),
                       nrow=4, normalize=False) # Automatically converts to uint8



        tend = time.time()

        timeEachEpoch = (tend - tstart)
        print('Time taken for training epoch {0} / {1} = {2:.1f} sec'.format(epoch+1,epochs,timeEachEpoch))





    # Post training, save the model weights for inference later on
    torch.save(decoder.state_dict(),"decoder_celeba.pth") # For generation/Inference, only Generator is required!









