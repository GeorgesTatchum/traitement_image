import cv2
import matplotlib.pyplot as plt
import numpy as np

# Charger l'image de premier plan et le masque alpha
foreground = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td5/images/02_personage.png"
)
alpha_mask = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td5/images/02_personage_alpha.png",
    cv2.IMREAD_GRAYSCALE,
)

# Convertir le masque alpha en flottant et l'étendre pour correspondre aux canaux RGB
alpha = alpha_mask.astype(float) / 255.0
alpha = cv2.merge([alpha, alpha, alpha])  # Étendre les dimensions pour RGB

# Obtenir les dimensions de l'image à incruster
h_fg, w_fg = foreground.shape[:2]


# Fonction pour incruster l'image sur un fond donné
def insert_foreground(background, foreground, alpha, x_offset, y_offset):
    """
    Incruste l'image foreground sur background avec un masque alpha, à la position (x_offset, y_offset).
    """
    h, w = alpha.shape[:2]
    background_cropped = background[y_offset : y_offset + h, x_offset : x_offset + w]

    # Multiplier foreground et background avec alpha et (1 - alpha)
    foreground_masked = foreground * alpha
    background_cropped_masked = background_cropped * (1 - alpha)

    # Combiner les deux
    blended_cropped = cv2.add(foreground_masked, background_cropped_masked)

    # Remplacer la région d'origine dans l'image de fond
    background[y_offset : y_offset + h, x_offset : x_offset + w] = blended_cropped
    return background


# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Position d'incrustation : Coordonnées de l'insertion
x_offset, y_offset = 50, 50  # Position en haut à gauche de l'image de la webcam

# Initialisation de Matplotlib
plt.ion()  # Mode interactif
fig, ax = plt.subplots(figsize=(8, 6))

# Lire les frames de la webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la frame vidéo.")
        break

    # Convertir la frame en float pour calculs
    frame = frame.astype(float) / 255.0

    # Vérifier si l'image à incruster ne dépasse pas les limites
    if x_offset + w_fg <= frame.shape[1] and y_offset + h_fg <= frame.shape[0]:
        # Insérer l'image de premier plan dans la vidéo
        frame = insert_foreground(frame, foreground / 255.0, alpha, x_offset, y_offset)

    # Convertir BGR -> RGB pour Matplotlib
    frame_rgb = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Affichage via Matplotlib
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis("off")
    ax.set_title("Incrustation en temps réel via Matplotlib")
    plt.pause(0.01)  # Pause pour mise à jour rapide

    # Quitter la boucle en appuyant sur 'q'
    if plt.waitforbuttonpress(timeout=0.01):
        break

# Libérer les ressources
cap.release()
plt.ioff()  # Désactiver le mode interactif
plt.close()
