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
from skimage import data, color
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Charge une image grayscale
img = image[:,:,0]
img = resize(img, (256, 256), anti_aliasing=True)

# FFT centrée
f = fftshift(fft2(img))

# Coordonnées spectrales
n, m = img.shape
Y, X = np.meshgrid(np.linspace(-1,1,m), np.linspace(-1,1,n))
theta = np.arctan2(Y, X)  # angle en radians

# Paramètres angulaires
n_angles = 8
angle_slices = np.linspace(-np.pi, np.pi, n_angles+1)

# Liste des composants directionnels
components = []

for i in range(n_angles):
    mask = (theta >= angle_slices[i]) & (theta < angle_slices[i+1])
    component = f * mask
    spatial_component = np.real(ifft2(ifftshift(component)))
    components.append(spatial_component)

# Affichage
fig, axs = plt.subplots(2, n_angles // 2, figsize=(15, 5))
for i in range(n_angles):
    ax = axs[i // (n_angles // 2)][i % (n_angles // 2)]
    ax.imshow(components[i], cmap='gray')
    ax.set_title(f"Angle {i}")
    ax.axis('off')

plt.suptitle("Décomposition spectrale directionnelle (approx. curvelets)")
plt.tight_layout()
plt.show()
