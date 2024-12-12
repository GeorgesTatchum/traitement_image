import cv2
import matplotlib.pyplot as plt
import numpy as np


def flou1(image):
    rows, cols = image.shape
    output = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            output[i, j] = (
                image[i, j]
                + image[i - 1, j]
                + image[i + 1, j]
                + image[i, j - 1]
                + image[i, j + 1]
            ) // 5

    return output


def flou2(image):
    rows, cols = image.shape
    output = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            output[i, j] = (
                image[i, j]
                + image[i - 1, j]
                + image[i + 1, j]
                + image[i, j - 1]
                + image[i, j + 1]
                + image[i - 1, j - 1]
                + image[i - 1, j + 1]
                + image[i + 1, j - 1]
                + image[i + 1, j + 1]
            ) // 9

    return output


def apply_flou2_three_times(image):
    for _ in range(3):
        image = flou2(image)
    return image


def blur_filtering(image_path, kernel_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
    return blurred_image


def gauss_filtering(image_path, kernel_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gaussian_blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return gaussian_blurred_image


def median_filtering(image_path, kernel_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    median_blurred_image = cv2.medianBlur(image, kernel_size)
    return median_blurred_image


def display_images(original_image, processed_image, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.show()


image_path = "C:/Users/georg/projets/m1/2025/trait_img/td1/images/lena.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
blurred_image = apply_flou2_three_times(original_image)
display_images(original_image, blurred_image, "Blurred Image")

blurred_image = blur_filtering(image_path, 5)
display_images(original_image, blurred_image, "Blurred Image with Kernel")

gaussian_blurred_image = gauss_filtering(image_path, 5)
display_images(original_image, gaussian_blurred_image, "Gaussian Blurred Image")

median_blurred_image = median_filtering(image_path, 5)
display_images(original_image, median_blurred_image, "Median Blurred Image")
