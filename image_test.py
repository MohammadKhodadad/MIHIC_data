import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = 'dataset/test/alveoli/0011-CD68-1-1.png'
image = Image.open(image_path)

image_rgb = image.convert('RGB')

# Convert the image to a numpy array
image_array = np.array(image_rgb)

# Save the image view
plt.imshow(image_array)
plt.title('Loaded Image')
plt.axis('off')  # Hide axes ticks
plt.savefig('loaded_image.png')  # Save the figure
plt.close()

# Show the image shape
print("Shape of the image:", image_array.shape)

# Save the distribution of pixel values
plt.figure(figsize=(10, 5))
plt.hist(image_array[:, :, 0].ravel(), bins=256, color='red', alpha=0.7, label='Red Channel')
plt.hist(image_array[:, :, 1].ravel(), bins=256, color='green', alpha=0.7, label='Green Channel')
plt.hist(image_array[:, :, 2].ravel(), bins=256, color='blue', alpha=0.7, label='Blue Channel')
plt.title('Pixel Value Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('pixel_distribution.png')  # Save the histogram figure
plt.close()