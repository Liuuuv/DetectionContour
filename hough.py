from contours import*

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data

# 1. Charger l'image et extraire les contours
edges = canny(image[:,:,0], sigma=2)  # Détection de contours avec Canny

# 2. Calcul de la transformée de Hough
hspace, angles, distances = hough_line(edges)

# 3. Détection des pics dans l'espace de Hough
peaks = hough_line_peaks(hspace, angles, distances, num_peaks=10)

# 4. Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Image originale
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Image originale')
axes[0].axis('off')

# Contours
axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Contours (Canny)')
axes[1].axis('off')

# Transformée de Hough
axes[2].imshow(np.log(1 + hspace), extent=[np.rad2deg(angles[-1]), np.rad2deg(angles[0]), 
                              distances[-1], distances[0]], cmap='jet', aspect='auto')
axes[2].set_xlabel('Angle (deg)')
axes[2].set_ylabel('Distance (pixels)')
axes[2].set_title('Espace de Hough')
# plt.colorbar(ax=axes[2])

# Affichage des lignes détectées sur l'image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap='gray')

for _, angle, dist in zip(*peaks):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax.plot((0, image.shape[1]), (y0, y1), '-r', linewidth=2)

ax.set_title('Lignes détectées')
ax.axis('off')
plt.tight_layout()
plt.show()