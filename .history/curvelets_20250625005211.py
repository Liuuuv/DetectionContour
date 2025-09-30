from contours import*

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from pyCurvelab import fdct_wrapping

# 1. Charger une image (exemple : coins de l'image "coins" de skimage)
image = color.rgb2gray(data.coins())  # Conversion en niveaux de gris
image = image[50:200, 50:200]  # Zoom sur une région avec contours

# 2. Paramètres de la transformée en curvelets
nb_scales = 4  # Nombre d'échelles
nb_angles = 16  # Nombre d'orientations par échelle

# 3. Appliquer la transformée
curvelets = fdct_wrapping(image, nb_scales=nb_scales, nb_angles=nb_angles)

# 4. Extraire les contours (coefficients des échelles fines)
edge_map = np.zeros_like(image)
for angle in range(len(curvelets[-2])):  # Échelle fine avant la dernière
    edge_map += np.abs(curvelets[-2][angle])  # Somme des coefficients

# 5. Normalisation et seuillage
edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
edge_map = (edge_map > 0.5).astype(np.uint8)  # Seuil binaire

# 6. Visualisation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Image Originale')
plt.subplot(1, 2, 2)
plt.imshow(edge_map, cmap='gray')
plt.title('Contours Détectés (Curvelets)')
plt.show()