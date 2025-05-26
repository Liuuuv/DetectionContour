from contours import*


img = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        advancement = i/(image.shape[0]-1)
        coef_dir = np.tan(np.pi * advancement / 2)
        line = []
        for x in np.linspace(0,image.shape[0]-1,10*image.shape[0]):
            if int(coef_dir*x+j) < image.shape[1] and not (int(x),int(coef_dir*x+j)) in line:
                line.append((int(x),int(coef_dir*x+j)))

        coef = np.sum([image[k,l][0] for k,l in line])
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



plot(image)
plot(img, cmap='magma')
plt.colorbar()

# original_ticks = plt.yticks()[0]
# new_labels = [f"{np.tan(np.pi * val / (2 * image.shape[0]-1)):.1f}" for val in original_ticks]  # Exemple de transformation
# plt.yticks(original_ticks, new_labels)
# plt.yscale('log')

plt.xlabel("ordonnée à l'origine")
plt.ylabel("coef directeur")
plt.title("repère tourné de pi/2 sens horaire")
plt.show()