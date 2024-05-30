import sys
sys.path.append('../')

import pandas as pd
from utils.dataloader import data_loader_files


import matplotlib.pyplot as plt
from PIL import Image
import os

data=data_loader_files()
addresses=data.addresses
addresses_splitted=addresses.apply(lambda x:x.replace('.png','').split('/')[-1].split('-'))
addresses=addresses[addresses_splitted.apply(lambda x:x[2] in ['1','2']  and x[3] in ['1','2'] and x[0] in ['0009'] and x[1] in ['CD38'])]
print(list(addresses.apply(lambda x:x.split('dataset/')[1])))

addresses=list(addresses)

n_images = len(addresses)
fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 8))  # Adjust size as needed

for ax, img_path in zip(axes, addresses):
    # Load the image
    img = Image.open(img_path)
    # Display the image
    ax.imshow(img)
    # Set the title as the file name
    ax.set_title(img_path.split('dataset/')[1])
    # Hide axes ticks
    ax.axis('off')

# Adjust layout
plt.tight_layout()
# Save the figure
plt.savefig('visualized_images.png')
# Optionally display the figure
plt.show()