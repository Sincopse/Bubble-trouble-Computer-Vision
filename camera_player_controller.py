import numpy as np

from camera import *


class CameraPlayerController:

    def __init__(self):
        self.cap = cv2.VideoCapture()
        self.image_hsv = None

        self.camera_1 = Camera()
        self.camera_2 = Camera()

        self.camera_1.open_camera("1")
        self.camera_2.open_camera("2")

    def update_camera(self):
        cv2.namedWindow("Camera Image")
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()

        image = image[:, ::-1, :]
        self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        camera_image_1 = self.camera_1.update_camera(self.image_hsv, 1)
        camera_image_2 = self.camera_2.update_camera(self.image_hsv, 2)
        camera_image = np.concatenate((camera_image_1, camera_image_2), 1)
        camera_image = cv2.resize(camera_image, (self.image_hsv.shape[1], int(self.image_hsv.shape[0]/2)))
        cv2.imshow("Camera Image", camera_image)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
