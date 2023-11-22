import numpy as np

from object_detection import *


class CameraPlayerController:

    def __init__(self):
        self.cap = cv2.VideoCapture()

        # Thresholds for both players
        self.camera_1 = ObjectDetection()
        self.camera_2 = ObjectDetection()

        self.isPlayer1Firing = False
        self.isPlayer2Firing = False
        self.player1Position = 0
        self.player2Position = 100

    def update_camera(self):
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()

        image = image[:, ::-1, :]

        (camera_image,
         position1, self.isPlayer1Firing,
         position2, self.isPlayer2Firing) = self.camera_1.process(
            image,
            67,
            0
         )

        if position1 != -1:
            self.player1Position = position1
        if position2 != -1:
            self.player2Position = position2

        cv2.imshow("Camera", camera_image)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
