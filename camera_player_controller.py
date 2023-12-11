import numpy as np

from object_detection import *
from tracker import *


class CameraPlayerController:

    def __init__(self):
        self.cap = cv2.VideoCapture()

        self.tracker_1 = Tracker()
        self.tracker_2 = Tracker()
        self.is_tracking = 0

        # Mouse Border Selector
        self.mouse_point_1 = None
        self.mouse_point_2 = None
        self.cropping = False

        self.isPlayer1Firing = False
        self.isPlayer2Firing = False
        self.player1Position = 0
        self.player2Position = 100

        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", self.click_and_crop)

    def get_object_box(self):
        if self.mouse_point_2 is not None and self.cropping is False:
            return (self.mouse_point_1[0], self.mouse_point_1[1],
                    self.mouse_point_2[0] - self.mouse_point_1[0],
                    self.mouse_point_2[1] - self.mouse_point_1[1])
        else:
            return None

    def click_and_crop(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_point_1 = (x, y)
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_point_2 = (x, y)
            self.cropping = False

        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            self.mouse_point_2 = (x, y)

    def update_camera(self):
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()

        image = image[:, ::-1, :]

        frame = image.copy()

        # Completely bad code, I'll try to fix later, for now it works
        box = self.get_object_box()
        if box is not None and self.is_tracking == 0:
            self.tracker_1.init_track(frame, box)
            self.is_tracking = 1
            self.mouse_point_1 = None
            self.mouse_point_2 = None
            self.cropping = False

        box = self.get_object_box()
        if box is not None and self.is_tracking == 1:
            self.tracker_2.init_track(frame, box)
            self.is_tracking = 2

        if self.is_tracking == 1:
            position1, self.isPlayer1Firing = self.tracker_1.track(frame)
            if position1 != -1:
                self.player1Position = position1

        elif self.is_tracking == 2:
            position1, self.isPlayer1Firing = self.tracker_1.track(frame)
            position2, self.isPlayer2Firing = self.tracker_2.track(frame)
            if position1 != -1:
                self.player1Position = position1
            if position2 != -1:
                self.player2Position = position2

        # End of terrible code

        if self.mouse_point_2 is not None:
            cv2.rectangle(frame, self.mouse_point_1, self.mouse_point_2, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
