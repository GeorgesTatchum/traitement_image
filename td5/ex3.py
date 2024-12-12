import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def apply_sepia(image):
    kernel = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    sepia_image = cv.transform(image, kernel)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image


def apply_negative(image):
    return cv.bitwise_not(image)


def apply_binary(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    return cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)


def apply_green_channel(image):
    green_channel = image.copy()
    green_channel[:, :, 0] = 0
    green_channel[:, :, 2] = 0
    return green_channel


cap = cv.VideoCapture("C:/Users/georg/projets/m1/2025/trait_img/td5/images/video2.avi")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Impossible de recevoir la frame (fin du flux?). Sortie ...")
        break

    height, width, _ = frame.shape
    part_height = height // 2
    part_width = width // 3

    parts = [
        frame[0:part_height, 0:part_width],  # Original
        frame[0:part_height, part_width : 2 * part_width],  # Negative
        frame[0:part_height, 2 * part_width : width],  # Binary
        frame[part_height:height, 0:part_width],  # Green Channel
        frame[part_height:height, part_width : 2 * part_width],  # Sepia
        frame[part_height:height, 2 * part_width : width],  # Pencil Sketch
    ]

    parts[1] = apply_negative(parts[1])
    parts[2] = apply_binary(parts[2])
    parts[3] = apply_green_channel(parts[3])
    parts[4] = apply_sepia(parts[4])
    _, parts[5] = cv.pencilSketch(parts[5], sigma_s=60, sigma_r=0.07, shade_factor=0.05)

    combined_image = np.vstack([np.hstack(parts[:3]), np.hstack(parts[3:])])

    plt.imshow(cv.cvtColor(combined_image, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.draw()
    plt.pause(0.001)
    plt.clf()

cap.release()
plt.close()
