from contours import*

def plot_filter_response(kernel, title="Réponse fréquentielle"):
    """
    Affiche le filtre et son spectre (module de la TF 2D).
    """
    # Calcul de la TF 2D
    fft_kernel = np.fft.fft2(kernel)
    fft_shifted = np.fft.fftshift(fft_kernel)  # Centrage
    magnitude = np.log(1 + np.abs(fft_shifted))  # Module en log

    # Affichage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filtre spatial
    ax1.imshow(kernel, cmap='gray')
    ax1.set_title("Filtre spatial")
    ax1.axis('off')
    
    # Spectre
    ax2.imshow(magnitude, cmap='viridis')
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.show()

filter = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
plot_filter_response(filter)