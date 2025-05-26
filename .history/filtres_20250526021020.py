from contours import*

def plot_filter_response(kernel, title="Réponse fréquentielle"):
    """
    Affiche le filtre et son spectre (module de la TF 2D).
    """
    # Calcul de la TF 2D
    fft_kernel = np.fft.fft2(kernel)
    fft_shifted = np.fft.fftshift(fft_kernel)  # Centrage
    magnitude = np.log(1 + np.abs(fft_shifted))  # Module en log

    
    h, w = magnitude.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]  # Version corrigée
    r = np.sqrt(x**2 + y**2).astype(int)
    
    # Utilisation de scipy.ndimage pour le profil radial
    radial_profile = np.mean(magnitude, labels=r, index=np.arange(0, max(h, w)//2))
    
    plt.figure()
    plt.plot(radial_profile)
    plt.xlabel("Distance du centre (fréquence)")
    plt.ylabel("Amplitude (log)")
    plt.title("Profil radial du spectre")
    plt.show()
    
    # # Affichage
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # # Filtre spatial
    # ax1.imshow(kernel, cmap='gray')
    # ax1.set_title("Filtre spatial")
    # ax1.axis('off')
    
    # # Spectre
    # ax2.imshow(magnitude, cmap='viridis')
    # ax2.set_title(title)
    # ax2.axis('off')
    
    # plt.show()

filter = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
plot_filter_response(filter)