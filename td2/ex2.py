import cv2
import matplotlib.pyplot as plt


def find_template_in_image(source_path, template_path):
    # Lire l'image source et l'image modèle
    source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

    # Convertir les images en niveaux de gris
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # Obtenir la largeur et la hauteur du modèle
    w, h = template_gray.shape[::-1]

    # Effectuer la correspondance de modèle en utilisant la corrélation croisée normalisée
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Obtenir l'emplacement de la meilleure correspondance
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Dessiner un rectangle autour de la région correspondante
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(source_img, top_left, bottom_right, (0, 255, 0), 2)

    # Afficher le résultat
    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    plt.title("Point détecté"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(result, cmap="gray")
    plt.title("Résultat de la correspondance"), plt.xticks([]), plt.yticks([])
    plt.show()


# Tester la fonction avec différents modèles et images sources
source_path = "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/source.jpg"
source_path2 = "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/source2.jpeg"
source_path3 = "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/source3.jpg"
template_path = (
    "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/template.jpg"
)
template_path2 = (
    "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/template2.jpeg"
)
template_path3 = (
    "C:/Users/georg/projets/m1/2025/trait_img/td2/images/moodle/template3.jpg"
)

find_template_in_image(source_path, template_path)
find_template_in_image(source_path, template_path2)
find_template_in_image(source_path, template_path3)
find_template_in_image(source_path2, template_path)
find_template_in_image(source_path3, template_path)
