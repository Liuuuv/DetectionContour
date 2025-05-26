from contours import*

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pywt import wavedec2
from scipy.fft import fft2, ifft2, fftshift

def curvelet_transform(img, n_scales=4):
    """
    Implémentation simplifiée d'une transformée en curvelets.
    """
    # Décomposition multi-échelle
    coeffs = wavedec2(img, 'db1', level=n_scales)
    
    # Initialisation du résultat
    curvelets = []
    
    # Traitement par échelle
    for i in range(1, n_scales+1):
        # Coefficients d'ondelettes à cette échelle
        cH, cV, cD = coeffs[i]
        
        # TF 2D des coefficients
        fft_cH = fftshift(fft2(cH))
        fft_cV = fftshift(fft2(cV))
        fft_cD = fftshift(fft2(cD))
        
        # Combinaison pour simuler des curvelets
        curvelet = np.abs(fft_cH) + np.abs(fft_cV) + np.abs(fft_cD)
        curvelets.append(curvelet)
    
    return coeffs[0], curvelets



# Application de la transformée
approx, curvelets = curvelet_transform(image, n_scales=4)

# Visualisation
plt.figure(figsize=(12, 8))

# Image originale
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')
plt.axis('off')

# Approximation
plt.subplot(2, 3, 2)
plt.imshow(approx, cmap='gray')
plt.title('Coeffs basse fréquence')
plt.axis('off')

# Curvelets par échelle
for i, curvelet in enumerate(curvelets, 3):
    plt.subplot(2, 3, i)
    plt.imshow(curvelet, cmap='jet', norm=LogNorm())
    plt.title(f'Échelle {i-2}')
    plt.axis('off')

plt.tight_layout()
plt.show()