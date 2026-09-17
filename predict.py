import cv2
import mediapipe as mp
import math
import os
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("asl_model.pkl")


# -----------------------------
# Load MediaPipe
# -----------------------------

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)


# -----------------------------
# Hand connections
# -----------------------------

connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# -----------------------------
# Main loop
# -----------------------------

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read webcam")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)


    # -----------------------------
    # If hand detected
    # -----------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        wrist = hand[0]


        # Find largest distance from wrist
        max_distance = 0

        for landmark in hand:

            dx = landmark.x - wrist.x
            dy = landmark.y - wrist.y

            distance = math.sqrt(
                dx**2 + dy**2
            )

            if distance > max_distance:
                max_distance = distance


        if max_distance != 0:

            # -----------------------------
            # Create input for model
            # -----------------------------

            row = []

            for landmark in hand:

                centered_x = (
                    landmark.x - wrist.x
                )

                centered_y = (
                    landmark.y - wrist.y
                )

                centered_z = (
                    landmark.z - wrist.z
                )


                scaled_x = (
                    centered_x / max_distance
                )

                scaled_y = (
                    centered_y / max_distance
                )

                scaled_z = (
                    centered_z / max_distance
                )


                row.extend([
                    scaled_x,
                    scaled_y,
                    scaled_z
                ])


            # -----------------------------
            # Make prediction
            # -----------------------------

            prediction = model.predict([row])[0]


            # -----------------------------
            # Display prediction
            # -----------------------------

            cv2.putText(
                frame,
                "Letter: " + prediction,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )


        # -----------------------------
        # Draw hand landmarks
        # -----------------------------

        for start, end in connections:

            x1 = int(
                hand[start].x * frame.shape[1]
            )

            y1 = int(
                hand[start].y * frame.shape[0]
            )

            x2 = int(
                hand[end].x * frame.shape[1]
            )

            y2 = int(
                hand[end].y * frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        for landmark in hand:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


    # -----------------------------
    # Show webcam
    # -----------------------------

    cv2.imshow(
        "ASL Recognition",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()