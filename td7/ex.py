import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons


def apply_sepia(image):
    kernel = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    sepia_image = cv2.transform(image, kernel)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image


def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    for i in range(overlay_resized.shape[0]):
        for j in range(overlay_resized.shape[1]):
            if overlay_resized[i, j, 3] != 0:
                background[y + i, x + j] = overlay_resized[i, j, :3]


def change_background(frame, background_image):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 20, 255))
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(background_image, background_image, mask=mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return cv2.add(bg, fg)


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


def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        "C:/Users/georg/projets/m1/2025/trait_img/td7/haarcascades/haarcascades/haarcascade_frontalface_alt.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        "C:/Users/georg/projets/m1/2025/trait_img/td7/haarcascades/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    )
    background_image = cv2.imread(
        "C:/Users/georg/projets/m1/2025/trait_img/td7/background.webp"
    )
    sunglasses = cv2.imread(
        "C:/Users/georg/projets/m1/2025/trait_img/td6/images/images/sunglasses.png",
        cv2.IMREAD_UNCHANGED,
    )
    background_image = cv2.resize(background_image, (640, 480))

    snowflakes = [
        {
            "x": random.randint(0, 640),
            "y": random.randint(-50, 480),
            "radius": random.randint(2, 5),
            "speed": random.randint(1, 3),
        }
        for _ in range(100)
    ]

    options = {
        "face_detection": False,
        "sepia": False,
        "background_change": False,
        "snowflakes": False,
        "check_all": False,
    }

    fig, (ax_video, ax_controls) = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [4, 1]}
    )
    plt.subplots_adjust(bottom=0.2)

    check_buttons = CheckButtons(
        ax_controls,
        ["Face Detection", "Sepia", "Background Change", "Snowflakes", "Check All"],
    )

    def update_options(label):
        options[label.lower().replace(" ", "_")] = not options[
            label.lower().replace(" ", "_")
        ]

    check_buttons.on_clicked(update_options)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if options["face_detection"] or options["check_all"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                frame = cv2.ellipse(
                    frame,
                    (x + int(w * 0.5), y + int(h * 0.5)),
                    (int(w * 0.5), int(h * 0.5)),
                    0,
                    0,
                    360,
                    (255, 255, 255),
                    4,
                )
                roi_gray = gray[y : y + h, x : x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:  # S'assurer que les deux yeux sont détectés
                    # Redimensionner les lunettes à la largeur du visage
                    resized_sunglasses = cv2.resize(
                        sunglasses,
                        (w, int(w * sunglasses.shape[0] / sunglasses.shape[1])),
                    )

                    # Obtenir la hauteur et la largeur de l'image des lunettes
                    sh, sw, sc = resized_sunglasses.shape

                    # Déterminer les coordonnées pour placer les lunettes
                    overlay_x = x
                    overlay_y = y + int(
                        h / 4
                    )  # Ajuster pour les positionner sur les yeux

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
                        frame[
                            overlay_y : overlay_y + sh, overlay_x : overlay_x + sw, c
                        ] = (
                            alpha_sunglasses * resized_sunglasses[:, :, c]
                            + alpha_background
                            * frame[
                                overlay_y : overlay_y + sh,
                                overlay_x : overlay_x + sw,
                                c,
                            ]
                        )

        if options["sepia"] or options["check_all"]:
            frame = apply_sepia(frame)

        if options["background_change"] or options["check_all"]:
            frame = change_background(frame, background_image)

        if options["snowflakes"] or options["check_all"]:
            frame = add_snowflakes(frame, snowflakes)

        ax_video.clear()
        ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_video.axis("off")
        plt.draw()
        plt.pause(0.001)

        if plt.waitforbuttonpress(0.001):
            break

    cap.release()
    plt.close()


if __name__ == "__main__":
    main()
