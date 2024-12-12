import cv2

message = "il n'existe pas vraiment"
binary_message = "".join(format(ord(char), "08b") for char in message)
print("binary_message", binary_message)

# Load the image
image_path = "C:/Users/georg/projets/m1/2025/trait_img/td3/images/02_cover.png"

cover = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE,
)

# convertir l'image en une liste de pixels
flat_image = cover.flatten()

# Ensure the message can fit in the image
if len(binary_message) > len(flat_image):
    raise ValueError("Message is too long to be hidden in the image.")

# mettre le message dans l'image
for i in range(len(binary_message)):
    bit = int(binary_message[i])
    if flat_image[i] % 2 == 0:
        if bit == 1:
            if flat_image[i] == 255:
                flat_image[i] -= 1
            else:
                flat_image[i] += 1
    else:
        if bit == 0:
            flat_image[i] -= 1

# Redimensionner l'image pour retrouver sa forme d'origine
cover_image = flat_image.reshape(cover.shape)

# Sauvegarder l'image tatouée
stego_cover_image_path = (
    "C:/Users/georg/projets/m1/2025/trait_img/td3/images/02_over_stego.png"
)
cv2.imwrite(stego_cover_image_path, cover_image)

# Extraction du message caché
extracted_bits = []
for i in range(len(binary_message)):
    if flat_image[i] % 2 == 0:
        extracted_bits.append("0")
    else:
        extracted_bits.append("1")

# Convertir les bits extraits en caractères
extracted_binary_message = "".join(extracted_bits)
extracted_message = "".join(
    chr(int(extracted_binary_message[i : i + 8], 2))
    for i in range(0, len(extracted_binary_message), 8)
)

print("extracted_message", extracted_message)
