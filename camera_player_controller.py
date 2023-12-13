import numpy as np

from tracker import *


class CameraPlayerController:

    def __init__(self):
        self.cap = cv2.VideoCapture()
        self.frame = None

        self.trackers = [Tracker(), Tracker()]

        self.current_tracker = False

        # Mouse Border Selector
        self.mouse_point_1 = None
        self.mouse_point_2 = None
        self.cropping = False

        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", self.click_and_crop)

        # Player Controlling Variables
        self.isPlayer1Firing = False
        self.isPlayer2Firing = False

        self.player1Position = 0
        self.player2Position = 100

    def get_box(self):
        return (self.mouse_point_1[0], self.mouse_point_1[1],
                self.mouse_point_2[0] - self.mouse_point_1[0],
                self.mouse_point_2[1] - self.mouse_point_1[1])

    def click_and_crop(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_point_1 = (x, y)
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.cropping = False
            self.mouse_point_2 = (x, y)

            if self.frame is not None:
                self.trackers[self.current_tracker].init_track(self.frame, self.get_box())
                self.current_tracker = not self.current_tracker

            self.mouse_point_1 = None
            self.mouse_point_2 = None

        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            cv2.rectangle(self.frame, self.mouse_point_1, (x, y), (0, 255, 0), 2)

    def update_camera(self):
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()

        image = image[:, ::-1, :]

        self.frame = image.copy()

        position1, self.isPlayer1Firing = self.trackers[0].track(self.frame)
        position2, self.isPlayer2Firing = self.trackers[1].track(self.frame)
        if position1 != -1:
            self.player1Position = position1
        if position2 != -1:
            self.player2Position = position2

        cv2.imshow("Camera", self.frame)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
