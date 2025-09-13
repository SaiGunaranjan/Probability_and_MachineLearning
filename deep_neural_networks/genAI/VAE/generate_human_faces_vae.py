# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:54:40 2025

@author: Sai Gunaranjan
"""

from vae_celebA import Decoder
import torch
import matplotlib.pyplot as plt
import os
import sys



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "decoder_celeba.pth"

if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found. Exiting...")
    sys.exit(1)   # exit gracefully

latent_space_dim = 128
decoder = Decoder(latent_space_dim) # Create object of Decoder arcitecture
decoder = decoder.to(device) # Move decoder object to devicde

# Load the model weights i.e decoder weights and mpa them to the Decoder object created
decoder.load_state_dict(torch.load(model_path,map_location=device))

decoder.eval() # Set decoder in eval mode

num_images_to_generate = 1

with torch.no_grad():
    latent_variable_z = torch.randn(num_images_to_generate,latent_space_dim) # Draw samples from a standard normal
    latent_variable_z = latent_variable_z.to(device)
    generatedImageFlatten = decoder(latent_variable_z)


generatedImage = generatedImageFlatten.reshape(num_images_to_generate,3,64,64)
generatedImage = generatedImage.cpu()
generatedImage = generatedImage.permute(0, 2, 3, 1)

# Display
plt.figure(1,figsize=(20,10))
plt.imshow(generatedImage[0,:,:,:])
plt.axis('off')
plt.show()

