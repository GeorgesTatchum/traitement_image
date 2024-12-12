import cv2
import matplotlib.pyplot as plt

image_path = "C:/Users/georg/projets/m1/2025/trait_img/td1/images/lena.jpg"
# Charger l'image en niveaux de gris
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Vérifier si l'image est chargée correctement
if image is None:
    print("Erreur lors du chargement de l'image.")
    exit()

# Inverser les couleurs de l'image
inverted_image = 255 - image

# Sauvegarder l'image obtenue
cv2.imwrite(image_path + "_inverted_image.jpg", inverted_image)

# Afficher l'image originale et l'image inversée
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Image Originale")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Image Inversée")
plt.imshow(inverted_image, cmap="gray")
plt.axis("off")

plt.show()

# 2- inversion en couleur

image_path = (
    "C:/Users/georg/projets/m1/2025/trait_img/td1/images/circles_in_a_circle.jpg"
)
# Charger l'image en couleur
image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Vérifier si l'image est chargée correctement
if image_color is None:
    print("Erreur lors du chargement de l'image en couleur.")
    exit()

# Inverser les couleurs de l'image en couleur
inverted_image_color = 255 - image_color

# Sauvegarder l'image obtenue
cv2.imwrite(image_path + "_inverted_image_color.jpg", inverted_image_color)

# Afficher l'image originale et l'image inversée en couleur
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Image Originale en Couleur")
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Image Inversée en Couleur")
plt.imshow(cv2.cvtColor(inverted_image_color, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
