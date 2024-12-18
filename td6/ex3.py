import cv2
import matplotlib.pyplot as plt

# Charger les classifieurs Haarcascade
face_cascade = cv2.CascadeClassifier(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/haarcascades/haarcascades/haarcascade_frontalface_alt.xml"
)

eye_cascade = cv2.CascadeClassifier(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/haarcascades/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
)

# Charger l'image des lunettes
sunglasses = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/images/images/sunglasses.png",
    cv2.IMREAD_UNCHANGED,
)  # Inclut canal alpha (transparence)

# Ouvrir la vidéo
video = cv2.VideoCapture(0)


# Fonction pour redimensionner et superposer les lunettes
def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))  # Redimensionner les lunettes
    for i in range(overlay_resized.shape[0]):
        for j in range(overlay_resized.shape[1]):
            if overlay_resized[i, j, 3] != 0:  # Vérifie la transparence (canal alpha)
                background[y + i, x + j] = overlay_resized[i, j, :3]


# Lire et afficher chaque image de la vidéo avec Matplotlib
fig, ax = plt.subplots()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for x, y, w, h in faces:
        # Détection des yeux dans le visage
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:  # Si au moins 2 yeux sont détectés
            # Ajuster la position et la taille des lunettes
            glasses_width = w
            glasses_height = int(h / 3)  # Ajuster la hauteur des lunettes
            y_offset = int(h / 3)  # Décaler vers le haut
            overlay_image(
                frame, sunglasses, x, y + y_offset, glasses_width, glasses_height
            )

    # Convertir l'image BGR en RGB pour Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Afficher l'image dans Matplotlib
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis("off")
    plt.pause(0.03)  # Pause pour créer l'effet vidéo
