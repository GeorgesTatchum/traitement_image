import cv2
import matplotlib.pyplot as plt
import numpy as np


def binarize_image(image_path, threshold):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded
    if image is None:
        print("Error: Image not found or unable to load.")
        return

    # Apply the threshold to binarize the image
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)

    # Display the original and binarized images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Binarized Image")
    plt.imshow(binary_image, cmap="gray")
    plt.axis("off")

    plt.show()


image_path = (
    "C:/Users/georg/projets/m1/2025/trait_img/td1/images/circles_in_a_circle.jpg"
)
binarize_image(image_path, 100)
binarize_image(image_path, 150)
binarize_image(image_path, 200)


def binarize_image_with_cv(image_path, threshold):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded
    if image is None:
        print("Error: Image not found or unable to load.")
        return

    # Apply the threshold to binarize the image using OpenCV
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Display the original and binarized images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Binarized Image with OpenCV")
    plt.imshow(binary_image, cmap="gray")
    plt.axis("off")

    plt.show()


binarize_image_with_cv(image_path, 100)
binarize_image_with_cv(image_path, 150)
binarize_image_with_cv(image_path, 200)
