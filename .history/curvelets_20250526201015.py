from contours import*

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from pywt import wavedec2
# from scipy.fft import fft2, ifft2, fftshift

# def curvelet_transform(img, n_scales=4):
#     """
#     Implémentation simplifiée d'une transformée en curvelets.
#     """
#     # Décomposition multi-échelle
#     coeffs = wavedec2(img[:,:,0], 'db1', level=n_scales)
    
#     # Initialisation du résultat
#     curvelets = []
    
#     # Traitement par échelle
#     for i in range(1, n_scales+1):
#         # Coefficients d'ondelettes à cette échelle
#         cH, cV, cD = coeffs[i]
        
#         # TF 2D des coefficients
#         fft_cH = fftshift(fft2(cH))
#         fft_cV = fftshift(fft2(cV))
#         fft_cD = fftshift(fft2(cD))
        
#         # Combinaison pour simuler des curvelets
#         curvelet = np.abs(fft_cH) + np.abs(fft_cV) + np.abs(fft_cD)
#         # curvelet = np.stack((curvelet,)*3, axis=-1).astype(np.float32)
#         curvelets.append(curvelet)
    
#     # coeffs[0] = np.stack((coeffs[0],)*3, axis=-1).astype(np.float32)
#     return coeffs[0], curvelets



# # Application de la transformée
# approx, curvelets = curvelet_transform(image, n_scales=4)

# # Visualisation
# plt.figure(figsize=(12, 8))

# # Image originale
# plt.subplot(2, 3, 1)
# plt.imshow(image)
# plt.title('Image originale')
# plt.axis('off')

# # Approximation
# plt.subplot(2, 3, 2)
# plt.imshow(approx)
# plt.title('Coeffs basse fréquence')
# plt.axis('off')

# # Curvelets par échelle
# for i, curvelet in enumerate(curvelets, 3):
#     plt.subplot(2, 3, i)
#     plt.imshow(curvelet, cmap='jet', norm=LogNorm())
#     plt.title(f'Échelle {i-2}')
#     plt.axis('off')

# plt.colorbar()
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from skimage.transform import radon
from pywt import dwt2

# 1. Image test (carré avec un cercle)
image = np.zeros((128, 128))
image[32:96, 32:96] = 1  # Carré central
y, x = np.ogrid[-64:64, -64:64]
image[x**2 + y**2 < 30**2] = 0.5  # Cercle

# 2. Décomposition en échelles (simulée par ondelettes)
coeffs = dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs

# 3. Transformée de Radon par échelle (simulation curvelet)
def curvelet_scale(data, angles=np.linspace(0, 180, 16, endpoint=False)):
    sinogram = radon(data, theta=angles, circle=False)
    # Application d'ondelettes 1D le long des projections (simplifié)
    return np.array([np.abs(fft2(proj)) for proj in sinogram.T]).T

# Calcul pour chaque "échelle" (ici 2 échelles simulées)
curvelet_coarse = curvelet_scale(cA)  # Échelle grossière
curvelet_fine = curvelet_scale(cH + cV + cD)  # Échelle fine

# 4. Visualisation
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')

plt.subplot(1, 3, 2)
plt.imshow(curvelet_coarse, cmap='jet', aspect='auto')
plt.title('Coeffs Curvelet (Échelle grossière)')
plt.xlabel('Angle')
plt.ylabel('Fréquence radiale')

plt.subplot(1, 3, 3)
plt.imshow(curvelet_fine, cmap='jet', aspect='auto')
plt.title('Coeffs Curvelet (Échelle fine)')
plt.xlabel('Angle')

plt.tight_layout()
plt.show()