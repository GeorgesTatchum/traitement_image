import cv2

# import matplotlib.pyplot as plt

stego = cv2.imread(
    "C:/Users/georg/projets/m1/2025/trait_img/td3/images/01_stego.png",
    cv2.IMREAD_GRAYSCALE,
)

# plt.imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
# plt.show()

print("stego", stego)

# modification des intensités des pixels

stego_cp = stego.copy()
stego_cp[stego_cp % 2 == 0] = 0
stego_cp[stego_cp % 2 != 0] = 1

print("stego_cp", stego_cp)

# Extraction du message caché
message_bits = []
for i in range(stego_cp.shape[0]):
    for j in range(stego_cp.shape[1]):
        message_bits.append(stego_cp[i, j])

# Regroupement des bits en octets
message_bytes = []

# prendre les 1176 premiers bits
for i in range(0, len(message_bits[:1176]), 8):
    byte = message_bits[i : i + 8]
    if len(byte) == 8:
        message_bytes.append(int("".join(map(str, byte)), 2))

# Conversion des octets en caractères ASCII
message = "".join([chr(byte) for byte in message_bytes])

print("Message caché:", message)

# Extraction du nombre caché dans le message
# piem = message.split()
# print("piem", piem)

# piem_clean = []
# for word in piem:
#     if "'" in word:
#         piem_clean += word.split("'")
#     else:
#         piem_clean.append(word)

# pi_digits = []

# for word in piem_clean:
#     if len(word) == 10:
#         pi_digits.append("0")
#     else:
#         pi_digits.append(str(len(word)))

# pi_number = "".join(pi_digits)
# print("Nombre caché:", pi_number)
