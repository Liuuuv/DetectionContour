from contours import*
import scipy

def plot_filter_response(kernel, title="Réponse fréquentielle"):
    """
    Affiche le filtre, son spectre et le profil radial.
    """
    # Calcul de la TF 2D
    fft_kernel = np.fft.fft2(kernel)
    fft_shifted = np.fft.fftshift(fft_kernel)  # Centrage
    magnitude = np.log(1 + np.abs(fft_shifted))  # Module en log

    # Affichage du filtre et spectre
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(kernel, cmap='gray')
    ax1.set_title("Filtre spatial")
    ax1.axis('off')
    ax2.imshow(magnitude, cmap='viridis')
    ax2.set_title(title)
    ax2.axis('off')
    plt.show()

    # Calcul du profil radial CORRIGÉ
    h, w = magnitude.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    r = np.sqrt(x**2 + y**2).astype(int)
    
    # Calcul du profil moyen
    max_radius = min(h, w) // 2
    radial_profile = np.zeros(max_radius)
    for radius in range(max_radius):
        mask = (r == radius)
        if np.any(mask):
            radial_profile[radius] = np.mean(magnitude[mask])
    
    # Affichage du profil radial
    plt.figure()
    plt.plot(radial_profile)
    plt.xlabel("Distance du centre (fréquence)")
    plt.ylabel("Amplitude (log)")
    plt.title("Profil radial du spectre")
    plt.grid(True)
    plt.show()

filter = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
plot_filter_response(filter)