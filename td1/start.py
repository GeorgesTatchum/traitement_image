import cv2
from matplotlib import pyplot as plt

# Read image as BGR image
img = cv2.imread("/path-to-image/circles_in_a_circle.jpg")


# Read image as grayscale image
# version 1
gray = cv2.imread("/path-to-image/circles_in_a_circle.jpg", cv2.IMREAD_GRAYSCALE)
# version 2
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show RGB image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.title("my picture rgb")
plt.show()

# Show grayscale image
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Gray using imread function")
plt.show()


plt.imshow(gray1, cmap="gray", vmin=0, vmax=255)
plt.title("Gray using cvtColor function")
plt.show()

# Write image in the file
cv2.imwrite("/path-to-image/circles_in_a_circle_gray.jpg", gray)

# image size
print("Image height = ", img.shape[0])
print("Image width = ", img.shape[1])

# Get access to a pixel
for i in range(0, 3):
    for j in range(0, 3):
        pixel = img[i, j]
        print("Pixel [", i, ", ", j, "]= ", pixel)

# Copy image
img_clone = img.copy()
