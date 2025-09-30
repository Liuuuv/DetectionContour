from contours import*

import numpy as np
import matplotlib.pyplot as plt
from pyCurvelab import fdct_wrapping

# 1. Charger une image (ex: contours synthétiques)
image = np.zeros((256, 256))
image[64:192, 64:192] = 1  # Carré blanc sur fond noir

# 2. Paramètres de la transformée
nb_scales = 4  # Nombre d'échelles
nb_angles = 16  # Nombre d'orientations par échelle

# 3. Appliquer la transformée
curvelets = fdct_wrapping(image, nb_scales=nb_scales, nb_angles=nb_angles)

# 4. Visualiser les coefficients
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Image Originale')

# Afficher un sous-ensemble de coefficients (ex: échelle 2)
scale_idx = 1
angle_idx = 0
coeffs = np.abs(curvelets[scale_idx][angle_idx])
axes[1].imshow(coeffs, cmap='hot', interpolation='nearest')
axes[1].set_title(f'Coefficients Curvelets (Échelle {scale_idx}, Angle {angle_idx})')
plt.show()