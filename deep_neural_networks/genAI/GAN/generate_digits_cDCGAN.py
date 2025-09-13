# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 17:31:27 2025

@author: Sai Gunaranjan
"""

import torch
from conditional_DC_GAN import Generator
import matplotlib.pyplot as plt
import os
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Below parameters must match training setup
z_dim = 100
num_class_labels= 10
embed_dim = 50

# --------------------------
# Check if model file exists
# --------------------------
model_path = "generator.pth"

if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found. Exiting...")
    sys.exit(1)   # exit gracefully

# Create an object of the generator class
gen = Generator(z_dim,num_class_labels,embed_dim)
# Move generator to device
gen = gen.to(device)
# Load trained Generator weights and move to device
gen.load_state_dict(torch.load(model_path,map_location=device))

# Run Generator in eval mode
gen.eval()

digit_to_generate = 7 # choose from [0,9]

with torch.no_grad():

    z = torch.randn(1,z_dim)
    z = z.to(device)

    class_labels_gen = torch.tensor([digit_to_generate])
    class_labels_gen = class_labels_gen.to(device)
    generatedImage = gen(z, class_labels_gen)
    generatedImageTransformed = ((generatedImage+1)/2)


    plt.imshow(generatedImage.squeeze().detach().cpu(),cmap='gray')