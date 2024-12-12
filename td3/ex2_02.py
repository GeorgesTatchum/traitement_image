import cv2

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
