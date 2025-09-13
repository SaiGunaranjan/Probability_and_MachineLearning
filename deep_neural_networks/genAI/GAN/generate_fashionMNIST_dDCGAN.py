# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 17:31:27 2025

@author: Sai Gunaranjan
"""

import torch
from conditional_DC_GAN_fashion_mnist import Generator
import matplotlib.pyplot as plt
import os
import sys


"""

| Label | Class Name  |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |
"""

class_labels = ['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']

classidx_to_generate = 9 # choose from [0,9]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Below parameters must match training setup
z_dim = 100
num_class_labels= 10
embed_dim = 50

# --------------------------
# Check if model file exists
# --------------------------
model_path = "generator_fashionmnist.pth"

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



with torch.no_grad():

    z = torch.randn(1,z_dim)
    z = z.to(device)

    class_labels_gen = torch.tensor([classidx_to_generate])
    class_labels_gen = class_labels_gen.to(device)
    generatedImage = gen(z, class_labels_gen)
    generatedImageTransformed = ((generatedImage+1)/2)

    plt.title(f"Generated image of {class_labels[classidx_to_generate]}")
    plt.imshow(generatedImage.squeeze().detach().cpu(),cmap='gray')