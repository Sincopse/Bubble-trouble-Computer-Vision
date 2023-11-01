import numpy as np

from camera_treshholders import *


class CameraPlayerController:

    def __init__(self):
        self.cap = cv2.VideoCapture()

        # CameraTreshholders for both players
        self.camera_1 = CameraTreshholders()
        self.camera_2 = CameraTreshholders()

        self.camera_1.open_camera("1")
        self.camera_2.open_camera("2")

    def update_camera(self, num):
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()

        image = image[:, ::-1, :]
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        camera_image_1 = self.camera_1.get_player_action_with_segmentation(image_hsv, num)
        camera_image_2 = self.camera_2.get_player_action_with_segmentation(image_hsv, num)
        camera_image = np.concatenate((camera_image_1, camera_image_2), 1)
        camera_image = cv2.resize(camera_image, (image_hsv.shape[1]*2, int(image_hsv.shape[0])))
        cv2.imshow("CameraTreshholders Image", camera_image)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
