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
from pycurvelet import CurveletTransform
from skimage.data import camera
from matplotlib.colors import LogNorm

# 1. Charger l'image et normaliser
image = camera().astype(float) / 255.0

# 2. Paramètres de la transformée
n_scales = 4          # Nombre d'échelles
n_angles = 16         # Nombre d'orientations par échelle
is_complex = True     # Curvelets complexes (module+phase)

# 3. Initialisation et calcul
curvelet = CurveletTransform(image.shape, n_scales, n_angles, is_complex)
coeffs = curvelet.fwd(image)  # Transformée directe

# 4. Visualisation des coefficients par échelle
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 2, 2)
for s in range(n_scales):
    scale_coeffs = np.sum(np.abs(coeffs[s]), axis=0)  # Somme sur les angles
    plt.imshow(scale_coeffs, cmap='jet', norm=LogNorm())
    plt.title(f'Coefficients Curvelet (Échelle {s+1})')
    plt.colorbar()
    plt.axis('off')
plt.tight_layout()
plt.show()