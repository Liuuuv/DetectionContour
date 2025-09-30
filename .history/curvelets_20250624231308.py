from contours import*

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycurve import Curvelet

def curvelet_transform_alternative(image_path):
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Initialize Curvelet transform
    curvelet = Curvelet(img_array)
    
    # Forward transform
    curvelet_coeffs = curvelet.fdct()
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(np.abs(curvelet_coeffs[0][0])), cmap='jet')
    plt.title('First Scale Curvelet Coefficients')
    plt.colorbar()
    
    plt.show()

# Usage
curvelet_transform_alternative("your_image.jpg")