# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:56:13 2025

@author: Sai Gunaranjan
"""

"""
1. Can move the upsampling operation inside the __init__ method and use nn.Upsample

"""


import sys
import os

# Get root directory (parent of DDPM)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)



import torch
from torchvision import transforms
from celeba_dataset import CelebADataset
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import torch.nn.functional as F
import time as time
import torch.optim as optim
from torchvision.utils import save_image







# Define transformation to be applied on the RGB images of faces
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])


# Download/load the dataset
dataset = CelebADataset(
    img_dir=r"D:\git\Probability_and_MachineLearning\deep_neural_networks\genAI\VAE\data_faces\img_align_celeba", # img_dir="./data_faces/img_align_celeba",
    transform=transform
)


batch_size = 16
# Define the dataloader to batch the data
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)


# Define where to run the code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saveImagesFolder = 'DDPM_generated_faces'
os.makedirs(saveImagesFolder, exist_ok=True)



# Define the U-Net architecture
class UNet(nn.Module):

    def __init__(self, embed_dim):

        super(UNet,self).__init__()

        self.embed_dim = embed_dim
        #%%

        #Encoder

        # Conv Block 1
        self.encblock1_conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), # Output: batchsize x 64 × 64 × 64
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU() # Swish and SiLU (Sigmoid Linear Unit) are the same. Swish/SiLU is a new type of activation fn introduced by google in 2017. It is x*sigmoid(x)
            )
        self.time_emb_encblock1 = nn.Linear(self.embed_dim, 64) # batchsize x 64, # Inject Time embedding: Map 128 --> numChannels (64)
        self.encblock1_conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), # Output: batchsize x 64×64×64
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU() # Swish and SiLU (Sigmoid Linear Unit) are the same. Swish/SiLU is a new type of activation fn introduced by google in 2017. It is x*sigmoid(x)
            )
        self.encblock1_conv2d_downsample = nn.Conv2d(64, 64, 4, 2, 1) # batchsize x 64 x 32 x 32


        # Conv Block 2
        self.encblock2_conv2d_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), # batchsize x 128 x 32 x 32
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU()
            )
        self.time_emb_encblock2 = nn.Linear(self.embed_dim, 128) # batchsize x 128, #Inject Time embedding: Map 128 --> numChannels (128)
        self.encblock2_conv2d_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), # batchsize x 128 x 32 x 32
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU()
            )
        self.encblock2_conv2d_downsample = nn.Conv2d(128, 128, 4, 2, 1) # batchsize x 128 x 16 x 16


        # Conv Block 3
        self.encblock3_conv2d_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), # batchsize x 256 x 16 x 16
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU()
            )
        self.time_emb_encblock3 = nn.Linear(self.embed_dim, 256) #Inject Time embedding: Map 128 --> numChannels (256)
        self.encblock3_conv2d_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), # batchsize x 256 x 16 x 16
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU()
            )
        self.encblock3_conv2d_downsample = nn.Conv2d(256, 256, 4, 2, 1) # batchsize x 256 x 8 x 8


        # Conv Block 4
        self.encblock4_conv2d_1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), # batchsize x 512 x 8 x 8
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU()
            )
        self.time_emb_encblock4 = nn.Linear(self.embed_dim, 512) #Inject Time embedding: Map 128 --> numChannels (512)
        self.encblock4_conv2d_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # batchsize x 512 x 8 x 8
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU()
            )
        self.encblock4_conv2d_downsample = nn.Conv2d(512, 512, 4, 2, 1) # batchsize x 512 x 4 x 4


        #%%

        # Bottleneck layer
        self.convblock_bottleneck_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # batchsize x 512 x 4 x 4
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU()
            )
        self.time_emb_bottleneck = nn.Linear(self.embed_dim, 512) #Inject Time embedding: Map 128 --> numChannels (512)
        self.convblock_bottleneck_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # batchsize x 512 x 4 x 4
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU()
            )


        #%%

        # Decoder

        # DeConv Block 1
        # Upsample the output of bottleneck layer
        self.decblock1_conv2d_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # batchsize x 512 x 8 x 8
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU()
            )
        self.time_emb_decblock1 = nn.Linear(self.embed_dim, 512) #Inject Time embedding: Map 128 --> numChannels (512)
        self.decblock1_conv2d_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # batchsize x 512 x 8 x 8
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU()
            )


        # DeConv Block 2
        # Upsample the output of Dec block 1
        self.decblock2_conv2d_1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), # batchsize x 256 x 16 x 16
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU()
            )
        self.time_emb_decblock2 = nn.Linear(self.embed_dim, 256) #Inject Time embedding: Map 128 --> numChannels (256)
        self.decblock2_conv2d_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), # batchsize x 256 x 16 x 16
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU()
            )


        # DeConv Block 3
        # Upsample the output of Dec block 2
        self.decblock3_conv2d_1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), # batchsize x 128 x 32 x 32
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU()
            )
        self.time_emb_decblock3 = nn.Linear(self.embed_dim, 128) #Inject Time embedding: Map 128 --> numChannels (128)
        self.decblock3_conv2d_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), # batchsize x 128 x 32 x 32
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU()
            )


        # DeConv Block 4
        # Upsample the output of Dec block 3
        self.decblock4_conv2d_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), # batchsize x 64 x 64 x 64
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU()
            )
        self.time_emb_decblock4 = nn.Linear(self.embed_dim, 64) #Inject Time embedding: Map 128 --> numChannels (64)
        self.decblock4_conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), # batchsize x 64 x 64 x 64
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU()
            )

        # Final layer, Assuming we are regressing over mu_theta or error epsilon.
        # But if we are regressing over X0 i.e input image, then there should be sigmoid after the final conv layer
        # because the input image has been normalized to [0,1]
        self.final_conv = nn.Conv2d(64, 3, 3, 1, 1) # 3 x 64 x 64




    def sinusoidal_embedding(self,time_index_batch):

        time_index_batch = time_index_batch.float()
        device = time_index_batch.device # Get the device where the algo is running
        batchsize = time_index_batch.shape[0]
        spe = torch.zeros(batchsize, self.embed_dim,device=device) # sinusoidal positional embedding
        i = torch.arange(self.embed_dim//2,device=device)
        freq = torch.exp((-math.log(10000.0) * (2*i)/self.embed_dim))
        spe[:,0::2] = torch.sin(time_index_batch[:,None] * freq[None,:])
        spe[:,1::2] = torch.cos(time_index_batch[:,None] * freq[None,:])


        """ Will crash if embed dim is not an even number"""

        return spe




    def forward(self, image_data, time_index_batch):

        # Obtain the sinusoidal embedding vector
        sinposembed = self.sinusoidal_embedding(time_index_batch)

        # Encoder

        # Conv Block 1
        conv_output = self.encblock1_conv2d_1(image_data)
        spe_mlp = self.time_emb_encblock1(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 64 x 1 x 1
        conv_output_encblock1 = self.encblock1_conv2d_2(conv_output)
        conv_output_downsample = self.encblock1_conv2d_downsample(conv_output_encblock1)

        # Conv Block 2
        conv_output = self.encblock2_conv2d_1(conv_output_downsample)
        spe_mlp = self.time_emb_encblock2(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 128 x 1 x 1
        conv_output_encblock2 = self.encblock2_conv2d_2(conv_output)
        conv_output_downsample = self.encblock2_conv2d_downsample(conv_output_encblock2)

        # Conv Block 3
        conv_output = self.encblock3_conv2d_1(conv_output_downsample)
        spe_mlp = self.time_emb_encblock3(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 256 x 1 x 1
        conv_output_encblock3 = self.encblock3_conv2d_2(conv_output)
        conv_output_downsample = self.encblock3_conv2d_downsample(conv_output_encblock3)


        # Conv Block 4
        conv_output = self.encblock4_conv2d_1(conv_output_downsample)
        spe_mlp = self.time_emb_encblock4(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 512 x 1 x 1
        conv_output_encblock4 = self.encblock4_conv2d_2(conv_output)
        conv_output_downsample = self.encblock4_conv2d_downsample(conv_output_encblock4)

        # Bottleneck layer
        conv_output = self.convblock_bottleneck_1(conv_output_downsample)
        spe_mlp = self.time_emb_bottleneck(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 512 x 1 x 1
        conv_output = self.convblock_bottleneck_2(conv_output)

        # Decoder

        # Deconv Block 1
        conv_output_upsample = F.interpolate(conv_output, scale_factor=2, mode="nearest")
        conv_output = self.decblock1_conv2d_1(conv_output_upsample)
        spe_mlp = self.time_emb_decblock1(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 512 x 1 x 1
        conv_output_decblock1 = self.decblock1_conv2d_2(conv_output)
        conv_output_decblock1 += conv_output_encblock4 # Skip/Residual connection. Add feature map of Enc Conv Block 4 to Dec Conv Block 1

        # Deconv Block 2
        conv_output_upsample = F.interpolate(conv_output_decblock1, scale_factor=2, mode="nearest")
        conv_output = self.decblock2_conv2d_1(conv_output_upsample)
        spe_mlp = self.time_emb_decblock2(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 256 x 1 x 1
        conv_output_decblock2 = self.decblock2_conv2d_2(conv_output)
        conv_output_decblock2 += conv_output_encblock3 # Skip/Residual connection. Add feature map of Enc Conv Block 3 to Dec Conv Block 2


        # Deconv Block 3
        conv_output_upsample = F.interpolate(conv_output_decblock2, scale_factor=2, mode="nearest")
        conv_output = self.decblock3_conv2d_1(conv_output_upsample)
        spe_mlp = self.time_emb_decblock3(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 128 x 1 x 1
        conv_output_decblock3 = self.decblock3_conv2d_2(conv_output)
        conv_output_decblock3 += conv_output_encblock2 # Skip/Residual connection. Add feature map of Enc Conv Block 2 to Dec Conv Block 3


        # Deconv Block 4
        conv_output_upsample = F.interpolate(conv_output_decblock3, scale_factor=2, mode="nearest")
        conv_output = self.decblock4_conv2d_1(conv_output_upsample)
        spe_mlp = self.time_emb_decblock4(sinposembed)
        conv_output += spe_mlp[:,:,None,None] #broadcast as batchsize x 64 x 1 x 1
        conv_output_decblock4 = self.decblock4_conv2d_2(conv_output)
        conv_output_decblock4 += conv_output_encblock1 # Skip/Residual connection. Add feature map of Enc Conv Block 1 to Dec Conv Block 4

        # Final output
        mu_theta = self.final_conv(conv_output_decblock4)

        return mu_theta



image_data_channels = 3
image_height = 64
image_width = 64

time_embed_dim = 128
num_time_steps = 1000
t_subset_size = 1#30
num_instances_epsilon = 1#10
beta = torch.linspace(1e-4, 2e-2, num_time_steps) # Linear noise schedules
alpha = (1 - beta).to(device)
alpha_bar = (torch.cumprod(alpha,dim=0)).to(device)


unet = UNet(time_embed_dim)
unet = unet.to(device)

# Define loss function objective
criterion = nn.MSELoss(reduction='sum') # because p_theta(x_t-1|xt) is a gaussian

optimizer = optim.Adam(unet.parameters(), lr=1e-4)

epochs = 30
numGeneratedImages = 9

if __name__ == "__main__":

    for epoch in range(epochs):
        tstart = time.time()

        unet.train()

        for batch_idx, image_data in enumerate(data_loader):

            internalBatchSize = image_data.shape[0]

            # Set gradients to 0
            optimizer.zero_grad()
            # Move data to device
            image_data = image_data.to(device)
            # Make multiple copies(num_instances_epsillon number of copies) of the input image for each example data in the batch
            image_data_expand = image_data.unsqueeze(1).expand(-1, num_instances_epsilon, -1, -1, -1) # X_0

            # Compute forward pass of the UNet for the denoising process

            # Consistency term or Denoise mathcing term
            t_subset = torch.randint(2, num_time_steps+1, (t_subset_size,), device=device, requires_grad=False) # Sample a subset of T to compute the denosing
            loss_consistency_term_allt = 0
            for t in t_subset:
                # Apply no grad here
                epsilon = torch.randn(internalBatchSize, num_instances_epsilon, image_data_channels, image_height, image_width, device=device) # Sample from standard normal
                X_t = torch.sqrt(alpha_bar[t-1])*image_data_expand + torch.sqrt(1 - alpha_bar[t-1])*epsilon

                # mu_q(xt,x0)
                mu_q = ((torch.sqrt(alpha[t-1]))*(1-alpha_bar[t-2])*X_t + \
                    torch.sqrt(alpha_bar[t-2])*(1-alpha[t-1])*image_data_expand)/(1-alpha_bar[t-1])

                # variance_q (t)
                variance_q_t = ((1-alpha[t-1])*(1-alpha_bar[t-2]))/(1-alpha_bar[t-1])

                #mu_theta(xt)
                X_t_flat = X_t.reshape(internalBatchSize*num_instances_epsilon, image_data_channels, image_height, image_width)
                time_index_batch = torch.tensor([t] * (internalBatchSize*num_instances_epsilon), device=device)

                mu_theta_xt = unet(X_t_flat, time_index_batch)
                mu_theta_xt = mu_theta_xt.reshape(internalBatchSize,num_instances_epsilon,image_data_channels, image_height, image_width)

                # Loss from consistency/denoising matching term. Mean should be taken only across batches and num_instances_epsilon. Not across the vector/matrix dimension!
                loss_consistency_term = criterion(mu_theta_xt,mu_q)/(internalBatchSize*num_instances_epsilon)
                loss_consistency_term = (1/(2*variance_q_t))*loss_consistency_term # There is a variance! (Ideally can be neglected as per paper)

                loss_consistency_term_allt += loss_consistency_term # Total loss of the denoising matching term for all the subsets of T

            # Reconstruction term
            # epsilon = torch.randn(internalBatchSize, num_instances_epsilon, image_data_channels, image_height, image_width, device=device) # Sample from standard normal
            # X_1 = torch.sqrt(alpha[0])*image_data_expand + torch.sqrt(1 - alpha[0])*epsilon
            # X_1_flat = X_1.reshape(internalBatchSize*num_instances_epsilon, image_data_channels, image_height, image_width)
            # time_index_batch = torch.tensor([1] * (internalBatchSize*num_instances_epsilon), device=device)

            # variance_q_1 = 1-alpha[0]

            # mu_theta_x1 = unet(X_1_flat, time_index_batch)
            # mu_theta_x1 = mu_theta_x1.reshape(internalBatchSize,num_instances_epsilon,image_data_channels, image_height, image_width)

            # loss_reconstruction_term = criterion(mu_theta_x1,image_data_expand)/(internalBatchSize*num_instances_epsilon)
            # loss_reconstruction_term = (1/(2*variance_q_1))*loss_reconstruction_term # There is a variance! (Ideally can be neglected as per paper)

            loss_ddpm = loss_consistency_term_allt #+ loss_reconstruction_term

            loss_ddpm.backward()

            optimizer.step()

        torch.cuda.empty_cache()

        #  Generate an image every N epochs
        if ((epoch+1)%1 == 0):
            print(f"Epoch [{epoch+1}/{epochs}], loss: {loss_ddpm.item():.4f}")
            # Set U-Net models in eval mode
            unet.eval()
            with torch.no_grad():
                X_t = torch.randn(numGeneratedImages, image_data_channels, image_height, image_width, device=device) # Sample from standard normal
                for t in reversed(torch.arange(1,num_time_steps+1)):
                    time_index_batch = torch.tensor([t] * numGeneratedImages, device=device)
                    mu_theta_xt = unet(X_t,time_index_batch)
                    epsilon = torch.randn(numGeneratedImages, image_data_channels, image_height, image_width, device=device) # Sample from standard normal
                    if t > 1:
                        variance_q_t = ((1-alpha[t-1])*(1-alpha_bar[t-2]))/(1-alpha_bar[t-1])
                    else:
                        variance_q_t = 1-alpha[0]

                    X_tminus1 = mu_theta_xt + torch.sqrt(variance_q_t)*epsilon
                    X_t = X_tminus1

                generatedImage = torch.clamp(X_t,0,1)
            generatedImage = generatedImage.cpu()
            save_image(generatedImage, saveImagesFolder + '//' + "generated_image_epoch_{}.png".format(epoch+1),
                       nrow=3, normalize=False) # Automatically converts to uint8


        tend = time.time()

        timeEachEpoch = (tend - tstart)
        print('Time taken for training epoch {0} / {1} = {2:.1f} sec'.format(epoch+1,epochs,timeEachEpoch))





