from __future__ import print_function

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

source_window = "Source image"
corners_window = "Coins détectés"
max_thresh = 255


def cornerHarris_demo(val):
    thresh = val
    # Paramètres du détecteur
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Détection des coins
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalisation
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Dessiner un cercle autour des coins
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
    # Afficher le résultat
    plt.figure(corners_window)
    plt.imshow(dst_norm_scaled, cmap="gray")
    plt.title(corners_window)
    plt.show()


# Charger l'image source et la convertir en niveaux de gris
image_path = "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/chessboard.png"
src = cv.imread(image_path, cv.IMREAD_COLOR)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Afficher l'image source en utilisant matplotlib
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
ax.set_title(source_window)

# Créer un curseur pour le seuil
ax_thresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
thresh_slider = Slider(ax_thresh, "Seuil", 0, max_thresh, valinit=200)


# Fonction de mise à jour pour le curseur
def update(val):
    thresh = int(thresh_slider.val)
    cornerHarris_demo(thresh)
    plt.draw()


thresh_slider.on_changed(update)

# Appel initial pour afficher les coins
cornerHarris_demo(200)

plt.show()
