import cv2
import matplotlib.pyplot as plt

image_bg = "C:/Users/georg/projets/m1/2025/trait_img/td5\images/01_background.jpg"
image_obj = "C:/Users/georg/projets/m1/2025/trait_img/td5\images/01_object.jpg"

background = cv2.imread(image_bg, cv2.IMREAD_COLOR)
object_img = cv2.imread(image_obj, cv2.IMREAD_COLOR)

# simplification du traitement de la soustraction en passant par le niveau de gris
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
object_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

# Soustraire l'arrière-plan de l'image de l'objet
foreground = cv2.absdiff(object_gray, background_gray)

ret, threeshold_forground = cv2.threshold(foreground, 10, 255, cv2.THRESH_BINARY)

# Afficher les images en utilisant matplotlib
plt.figure(figsize=(12, 5))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(foreground, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(threeshold_forground, cmap="gray")
plt.axis("off")

plt.show()
