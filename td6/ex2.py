import cv2
import matplotlib.pyplot as plt

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/haarcascades/haarcascades/haarcascade_frontalface_alt.xml"
)

eye_cascade = cv2.CascadeClassifier(
    "C:/Users/georg/projets/m1/2025/trait_img/td6/haarcascades/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
)

video_path = "C:/Users/georg/projets/m1/2025/trait_img/td6/images/images/video1.mp4"
video_path_2 = "C:/Users/georg/projets/m1/2025/trait_img/td6/images/images/video2.mp4"


def detect_faces(image):
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw circle around the face and detect eyes within the face region
    if len(faces) > 0:
        for x, y, w, h in faces:
            image = cv2.ellipse(
                image,
                (x + int(w * 0.5), y + int(h * 0.5)),
                (int(w * 0.5), int(h * 0.5)),
                0,
                0,
                360,
                (255, 0, 255),
                4,
            )
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = image[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                center = (ex + ew // 2, ey + eh // 2)
                axes = (ew // 2, eh // 2)
                cv2.ellipse(roi_color, center, axes, 0, 0, 360, (255, 0, 0), 2)
    return image


def detect_on_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Impossible de recevoir la frame (fin du flux?). Sortie ...")
            break

        frame = detect_faces(frame)

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    cap.release()
    cv2.destroyAllWindows()


detect_on_video(video_path)
