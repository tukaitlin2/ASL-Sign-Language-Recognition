import csv
import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os

csv_filename = "asl_data.csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["label"]

        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])

        writer.writerow(header)

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path = model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

connections = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]

while cap.isOpened():

    success,frame = cap.read()
    if not success:
        print("Could not read webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
       image_format = mp.ImageFormat.SRGB,
       data = rgb_frame 
    )

    result = detector.detect(mp_image)


    for hand in result.hand_landmarks:
        for start, end in connections:

            x1 = int(hand[start].x * frame.shape[1])
            y1 = int(hand[start].y * frame.shape[0])

            x2 = int(hand[end].x * frame.shape[1])
            y2 = int(hand[end].y * frame.shape[0])

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame, 
                (x,y),
                5,
                (0,255,0),
                -1
            )

    cv2.imshow("Hand Landmark Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"):
        break

    if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wrist = hand[0]
    
            max_distance = 0
    
            for landmark in hand:
                dx = landmark.x - wrist.x
                dy = landmark.y - wrist.y
    
                distance = math.sqrt(dx**2 + dy**2)
    
                if distance > max_distance:
                    max_distance = distance
    
            if key == ord("v"):
                print("A key detected!")
                if max_distance == 0:
                    print("Could not scale hand")
                else:
                    label = "G"
                    row = [label]
    
                    for landmark in hand:
                        centered_x = landmark.x - wrist.x
                        centered_y = landmark.y - wrist.y
                        centered_z = landmark.z - wrist.z
    
                        scaled_x = centered_x / max_distance
                        scaled_y = centered_y / max_distance
                        scaled_z = centered_z / max_distance
    
                        row.extend([
                        scaled_x,
                        scaled_y,
                        scaled_z
                        ])

                    print("Saving data...")

                    with open(csv_filename, "a", newline = "") as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
    
                    print("Saved centered A!")

cap.release()
cv2.destroyAllWindows()