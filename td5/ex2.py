import cv2
import matplotlib.pyplot as plt

# Lire les images
background = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td5/images/02_background.jpg"
)
foreground = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td5/images/02_personage.png",
)
alpha_mask = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td5/images/02_personage_alpha.png",
    cv2.IMREAD_GRAYSCALE,
)

# Étape 2 : Convertir les valeurs uint8 en flottant
alpha = alpha_mask.astype(float) / 255.0
alpha = cv2.merge([alpha, alpha, alpha])  # Expand dimensions to match foreground

# Étape 3 : Extraire une région du background correspondant à la taille d'alpha
h, w = alpha.shape[0], alpha.shape[1]
background_cropped = background[:h, :w]  # Extraire la région correspondante

foreground_masked = foreground * alpha

# Étape 5 : Multiplier la région du fond par (1 - alpha)
background_cropped_masked = background_cropped * (1.0 - alpha[:, :, :3])

# Étape 6 : Ajouter le premier plan et l'arrière-plan masqués
blended_cropped = cv2.add(foreground_masked, background_cropped_masked)

# Étape 7 : Réinsérer la région fusionnée dans l'image de fond initiale
result = background.copy()
result[:h, :w] = blended_cropped

# Convertir l'image de BGR à RGB pour l'affichage avec matplotlib
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Étape 8 : Afficher l'image avec matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(result_rgb)
plt.axis("off")
plt.title("Incrustation de l'image")
plt.show()
