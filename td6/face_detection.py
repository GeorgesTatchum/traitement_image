# A face detection

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/haarcascades/haarcascades/haarcascade_frontalface_alt.xml"
)
# Read the input image
img = cv2.imread("C:/Users/georg/projets/m1/2025/trait_img/td6/images/images/one.jpg")

img_multi = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/images/images/multiple.jpg"
)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_multi = cv2.cvtColor(img_multi, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
faces_multi = face_cascade.detectMultiScale(gray_multi, 1.1, 4)

print(f"face len: {len(faces)}")
print(f"faces len: {len(faces_multi)}")

# Draw circle around the face
img = cv2.ellipse(
    img,
    (faces[0, 0] + int(faces[0, 2] * 0.5), faces[0, 1] + int(faces[0, 3] * 0.5)),
    (int(faces[0, 2] * 0.5), int(faces[0, 3] * 0.5)),
    0,
    0,
    360,
    (255, 0, 255),
    4,
)

# Draw circle around the faces
for i in range(faces_multi.shape[0]):
    img_multi = cv2.ellipse(
        img_multi,
        (
            faces_multi[i, 0] + int(faces_multi[i, 2] * 0.5),
            faces_multi[i, 1] + int(faces_multi[i, 3] * 0.5),
        ),
        (int(faces_multi[i, 2] * 0.5), int(faces_multi[i, 3] * 0.5)),
        0,
        0,
        360,
        (255, 0, 255),
        4,
    )


# Show RGB image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_multi = cv2.cvtColor(img_multi, cv2.COLOR_BGR2RGB)

plt.imshow(rgb)
plt.title("Detected face")
plt.show()

plt.imshow(rgb_multi)
plt.title("Detected faces")
plt.show()
