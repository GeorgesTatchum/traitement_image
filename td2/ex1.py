"""Exercice 1"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

rows = 2
columns = 2
seuils = [64, 100, 150, 200]


""" Binarise une image au seuil défini """


def seuillage(image, seuil):
    seuil, img = cv2.threshold(image, seuil, 255, cv2.THRESH_BINARY)
    return img


def erosion(image):
    img = image.copy()

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            img[i, j] = (
                0
                if image[i, j] == 0
                and image[i - 1, j] == 0
                and image[i - 1, j + 1] == 0
                and image[i, j + 1] == 0
                and image[i + 1, j + 1] == 0
                and image[i + 1, j] == 0
                and image[i + 1, j - 1] == 0
                and image[i, j - 1] == 0
                and image[i - 1, j - 1] == 0
                else 255
            )

    return img


def dilatation(image):
    img = image.copy()

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            img[i, j] = (
                0
                if image[i, j] == 0
                or image[i - 1, j] == 0
                or image[i - 1, j + 1] == 0
                or image[i, j + 1] == 0
                or image[i + 1, j + 1] == 0
                or image[i + 1, j] == 0
                or image[i + 1, j - 1] == 0
                or image[i, j - 1] == 0
                or image[i - 1, j - 1] == 0
                else 255
            )

    return img


# Seuillage de morph_1
morph_1 = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/morph_1.pgm",
    cv2.IMREAD_GRAYSCALE,
)
fig = plt.figure(figsize=(12, 8))

images_seuilles = []

for i, seuil in enumerate(seuils):
    fig.add_subplot(rows, columns, i + 1)
    img = seuillage(morph_1, seuil)
    images_seuilles.append(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"(Seuillage {seuil} morph_1)")

plt.show()

fig2 = plt.figure(figsize=(12, 8))
fig2.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(erosion(images_seuilles[-1]), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("(Erosion morph_1)")

fig2.add_subplot(rows, columns, 2)
plt.imshow(cv2.cvtColor(dilatation(images_seuilles[-1]), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("(Dilatation morph_1)")

plt.show()

""" Calcul du gradient morphologique"""

banane = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/banane.jpg",
    cv2.IMREAD_GRAYSCALE,
)


def gradient_morphologique(image):
    eroded = cv2.erode(image, None)
    dilated = cv2.dilate(image, None)
    gradient = cv2.absdiff(dilated, eroded)
    return gradient


gradient = gradient_morphologique(banane)

fig3 = plt.figure(figsize=(12, 8))
fig3.add_subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("(Gradient Morphologique banane)")

plt.show()
