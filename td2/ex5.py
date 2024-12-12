import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Fonction de transformation photomaton
def photomaton_transform(image):
    n, m = image.shape
    transformed_image = np.zeros_like(image)

    # Parcours de chaque pixel de l'image
    for i in range(n):
        for j in range(m):
            # Calcul des nouvelles coordonnées après transformation
            i_prime = (i // 2) + (i % 2) * (n // 2)
            j_prime = (j // 2) + (j % 2) * (m // 2)
            transformed_image[i_prime, j_prime] = image[i, j]

    return transformed_image


if __name__ == "__main__":
    # Création d'une image 256x256 avec des valeurs aléatoires
    # image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    # Chargement de l'image
    image_path = (
        "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/chessboard.png"
    )
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)

    # S'assurer que l'image a un nombre pair de lignes et de colonnes
    n, m = image.shape
    if n % 2 != 0:
        image = image[:-1, :]
    if m % 2 != 0:
        image = image[:, :-1]

    # Application de la transformation photomaton
    transformed_image = photomaton_transform(image)

    # Affichage des images originale et transformée
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image Originale")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Image Transformée")
    plt.imshow(transformed_image, cmap="gray")

    plt.show()
