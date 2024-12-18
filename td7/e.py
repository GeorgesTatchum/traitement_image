import random

import cv2
import numpy as np

# Charger les classificateurs Haar
face_cascade_path = "C:/Users/sorus/OneDrive/Documents/Python Scripts/TD6/haarcascades/haarcascade_frontalface_alt.xml"
eye_cascade_path = "C:/Users/sorus/OneDrive/Documents/Python Scripts/TD6/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


cap = cv2.VideoCapture(0)

# Charger l'image des lunettes
sunglasses_path = (
    "C:/Users/sorus/OneDrive/Documents/Python Scripts/TD6/images/sunglasses.png"
)
sunglasses = cv2.imread(
    sunglasses_path, cv2.IMREAD_UNCHANGED
)  # Inclure l'alpha (transparence)

mask_path = "C:/Users/sorus/OneDrive/Documents/Python Scripts/TD7/hacker2.webp"
mask = cv2.imread(
    mask_path, cv2.IMREAD_UNCHANGED
)  # Lecture avec transparence (canal alpha)


if sunglasses is None:
    print(f"Erreur : Impossible de charger l'image {sunglasses_path}")
    exit()


def apply_sepia(frame):
    # Matrice de transformation pour le filtre sépia
    sepia_filter = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    # Appliquer le filtre sur l'image
    sepia_frame = cv2.transform(frame, sepia_filter)
    sepia_frame = np.clip(sepia_frame, 0, 255)  # Éviter les débordements de pixels
    return sepia_frame.astype(np.uint8)


def add_snowflakes(frame, snowflakes):
    for snowflake in snowflakes:
        cv2.circle(
            frame,
            (snowflake["x"], snowflake["y"]),
            snowflake["radius"],
            (255, 255, 255),
            -1,
        )
        snowflake["y"] += snowflake["speed"]
        if snowflake["y"] > frame.shape[0]:
            snowflake["y"] = random.randint(-50, -10)
            snowflake["x"] = random.randint(0, frame.shape[1])
    return frame


snowflakes = [
    {
        "x": random.randint(0, 640),
        "y": random.randint(-50, 480),
        "radius": random.randint(2, 5),
        "speed": random.randint(1, 3),
    }
    for _ in range(100)
]


def overlay_image(background, overlay, x, y, width, height):
    overlay = cv2.resize(
        overlay, (width, height)
    )  # Redimensionne l'image d'incrustation
    for i in range(overlay.shape[0]):
        for j in range(overlay.shape[1]):
            if overlay[i, j, 3] != 0:  # Si le pixel n'est pas transparent
                background[y + i, x + j] = overlay[i, j, :3]  # Remplace les pixels
    return background


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris pour la détection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for x, y, w, h in faces:
        # Détecter les yeux dans la région du visage
        face_roi_gray = gray_frame[y : y + h, x : x + w]
        face_roi_color = frame[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30)
        )
        frame = overlay_image(frame, mask, x, y, w, h)

        if len(eyes) >= 2:  # S'assurer que les deux yeux sont détectés
            # Redimensionner les lunettes à la largeur du visage
            resized_sunglasses = cv2.resize(
                sunglasses, (w, int(w * sunglasses.shape[0] / sunglasses.shape[1]))
            )

            # Obtenir la hauteur et la largeur de l'image des lunettes
            sh, sw, sc = resized_sunglasses.shape

            # Déterminer les coordonnées pour placer les lunettes
            overlay_x = x
            overlay_y = y + int(h / 4)  # Ajuster pour les positionner sur les yeux

            # S'assurer que les lunettes restent dans les limites de l'image
            if overlay_y + sh > frame.shape[0]:
                sh = frame.shape[0] - overlay_y
                resized_sunglasses = resized_sunglasses[:sh, :, :]
            if overlay_x + sw > frame.shape[1]:
                sw = frame.shape[1] - overlay_x
                resized_sunglasses = resized_sunglasses[:, :sw, :]

            # Extraire le canal alpha (transparence) des lunettes
            alpha_sunglasses = resized_sunglasses[:, :, 3] / 255.0
            alpha_background = 1.0 - alpha_sunglasses

            # Superposer les lunettes sur l'image
            for c in range(0, 3):  # Pour les canaux BGR
                frame[overlay_y : overlay_y + sh, overlay_x : overlay_x + sw, c] = (
                    alpha_sunglasses * resized_sunglasses[:, :, c]
                    + alpha_background
                    * frame[overlay_y : overlay_y + sh, overlay_x : overlay_x + sw, c]
                )

    # Appliquer le filtre sépia
    frame = apply_sepia(frame)

    frame = add_snowflakes(frame, snowflakes)

    # Afficher le flux vidéo modifié
    cv2.imshow("Filtre Sépia", frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
