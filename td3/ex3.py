import random

import cv2
import numpy as np


def binarize_image(image):
    """Convertir l'image en noir et blanc."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def create_shares(image):
    """
    Créer deux masques pour l'image binaire donnée.
    Args:
        image (numpy.ndarray): Une image binaire 2D (valeurs 0 ou 1) de type uint8.
    Returns:
        tuple: Deux masques (share1, share2) de type numpy.ndarray, chacun de taille (2*rows, 2*cols) et de type uint8.
    """
    """Créer deux masques pour l'image binaire donnée."""
    rows, cols = image.shape
    share1 = np.zeros((rows * 2, cols * 2), dtype=np.uint8)
    share2 = np.zeros((rows * 2, cols * 2), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            patterns = [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ]
            pattern = random.choice(patterns)
            if pixel == 0:  # Pixel noir
                share1[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = pattern
                share2[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = pattern
            else:  # Pixel blanc
                share1[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = pattern
                share2[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = 1 - np.array(pattern)

    return share1 * 255, share2 * 255


def save_image(array, filename):
    """Enregistrer un tableau numpy en tant qu'image."""
    cv2.imwrite(filename, array)


def vcs(nom_fichier_image: str):
    """Schéma de cryptographie visuelle pour diviser l'image en deux masques."""
    image = cv2.imread(nom_fichier_image)
    binary_image = binarize_image(image)
    share1, share2 = create_shares(binary_image)
    save_image(share1, nom_fichier_image + "_1.png")
    save_image(share2, nom_fichier_image + "_2.png")


def decode_vcs(share1_path, share2_path):
    """Décoder deux masques pour retrouver l'image originale."""
    share1 = cv2.imread(share1_path, cv2.IMREAD_GRAYSCALE)
    share2 = cv2.imread(share2_path, cv2.IMREAD_GRAYSCALE)
    decoded_image = cv2.bitwise_or(share1, share2)
    save_image(decoded_image, "decoded_image.png")


image_path = "C:/Users/georg/projets/m1/2025/trait_img/td3/images/02_cover.png"


# # vcs(nom_fichier_image=image_path)
decode_vcs("_1.png", "_2.png")
