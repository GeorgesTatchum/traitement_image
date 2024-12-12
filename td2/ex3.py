import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/lena.jpg"
# Charger l'image en niveaux de gris
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calculer le gradient en direction x
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Calculer le gradient en direction y
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculer la magnitude du gradient
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normaliser la magnitude pour l'affichage
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convertir la magnitude en type de données 8 bits
gradient_magnitude = np.uint8(gradient_magnitude)

# Afficher les résultats
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Gradient X")
plt.imshow(sobel_x, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Gradient Y")
plt.imshow(sobel_y, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap="gray")
plt.axis("off")

plt.show()
