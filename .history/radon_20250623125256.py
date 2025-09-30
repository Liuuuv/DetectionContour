from contours import*

def radon_naive_cartesian():
    img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            advancement = i/(image.shape[0]-1)
            coef_dir = np.tan(np.pi * advancement / 2)
            line = []
            for x in np.linspace(0,image.shape[0]-1,10*image.shape[0]):
                if int(coef_dir*x+j) < image.shape[1] and not (int(x),int(coef_dir*x+j)) in line:
                    line.append((int(x),int(coef_dir*x+j)))

            coef = np.sum([image[k,l] for k,l in line])/image.shape[0]
            img[i,j] += coef

    # img /= np.max(img)


    # i,j = 0, 0
    # line = []
    # for x in np.linspace(0,image.shape[0]-1,10*image.shape[0]):
    #     if int(i*x+j) < image.shape[1] and not (int(x),int(i*x+j)) in line:
    #         line.append((int(x),int(i*x+j)))
    # for k,l in line:
    #     img[k,l] = 1

    # print(np.sum([image[k,l] for k,l in line]))



    # plot(image)
    # plot(img)
    plt.imshow(img)

    # original_ticks = plt.yticks()[0]
    # new_labels = [f"{np.tan(np.pi * val / (2 * image.shape[0]-1)):.1f}" for val in original_ticks]  # Exemple de transformation
    # plt.yticks(original_ticks, new_labels)
    # plt.yscale('log')

    plt.xlabel("ordonnée à l'origine")
    plt.ylabel("coef directeur")
    plt.title("repère tourné de pi/2 sens horaire")
    plt.show()


# radon_naive_cartesian()


from skimage.transform import radon

image_ = image[:,:,0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image)

theta = np.linspace(0.0, 180.0, max(image_.shape), endpoint=False)
sinogram = radon(image_, theta=theta)
dx, dy = 0.5 * 180.0 / max(image_.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Sinogramme")
ax2.set_xlabel("Angle")
ax2.set_ylabel("Position")
ax2.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect='auto',
)

fig.tight_layout()
plt.show()