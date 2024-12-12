import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "C:/Users/georg/projets/m1/2025/trait_img/td1/images/lena.jpg"

lena = cv2.imread(image_path)
b, g, r = cv2.split(lena)
imgv = g
imgb = b
imgr = r

# Affichage en niveaux de gris
fig_gray, axs_gray = plt.subplots(1, 3, figsize=(15, 5))
axs_gray[0].imshow(imgv, cmap="gray", vmin=0, vmax=255)
axs_gray[0].set_title("Canal vert (niveau de gris)")
axs_gray[0].axis("off")

axs_gray[1].imshow(imgb, cmap="gray", vmin=0, vmax=255)
axs_gray[1].set_title("Canal bleu (niveau de gris)")
axs_gray[1].axis("off")

axs_gray[2].imshow(imgr, cmap="gray", vmin=0, vmax=255)
axs_gray[2].set_title("Canal rouge (niveau de gris)")
axs_gray[2].axis("off")

plt.show()

# Affichage en couleur des canaux vert, bleu et rouge
taille = lena.shape[0], lena.shape[1]
nulle = np.zeros(taille, dtype=np.uint8)
imgv = cv2.merge((nulle, g, nulle))
imgb = cv2.merge((nulle, nulle, b))
imgr = cv2.merge((r, nulle, nulle))

fig_color, axs_color = plt.subplots(1, 3, figsize=(15, 5))
axs_color[0].imshow(imgv)
axs_color[0].set_title("Canal vert (couleurs)")
axs_color[0].axis("off")

axs_color[1].imshow(imgb)
axs_color[1].set_title("Canal bleu (couleurs)")
axs_color[1].axis("off")

axs_color[2].imshow(imgr)
axs_color[2].set_title("Canal rouge (couleurs)")
axs_color[2].axis("off")

plt.show()

# Assemblage de canaux
circles_in_a_circle = cv2.imread("images/circles_in_a_circle.jpg")
b, g, r = cv2.split(circles_in_a_circle)

# Les ronds rouges deviennent verts
# Les formes jaunes (rouge + vert) deviennent bleu-vert. Or, le vert a remplacé le rouge donc le bleu remplace le vert
# Les formes bleu-vert deviennent violettes (rouge + bleu). Or, le bleu a remplacé le vert donc le rouge remplace le bleu
img = cv2.merge([g, r, b])

plt.imshow(img)
plt.title("Assemblage")
plt.axis("off")
plt.show()
