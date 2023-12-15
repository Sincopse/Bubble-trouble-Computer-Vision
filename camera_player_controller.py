from tracker import *


class CameraPlayerController:

    def __init__(self):
        self.cap = cv2.VideoCapture()
        self.cap.open(0)
        self.frame = None

        self.trackers = [Tracker(), Tracker()]

        self.current_tracker = False

        # Mouse Border Selector
        self.mouse_point_1 = None
        self.mouse_point_2 = None

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

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_point_2 = (x, y)

            if self.frame is not None:
                self.trackers[self.current_tracker].init_track(self.frame, self.get_box())
                self.current_tracker = not self.current_tracker

            self.mouse_point_1 = None
            self.mouse_point_2 = None

        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_point_1:
            self.mouse_point_2 = (x, y)

    def update_camera(self):
        _, image = self.cap.read()

        image = image[:, ::-1, :]

        self.frame = image.copy()

        self.player1Position, self.isPlayer1Firing = self.trackers[0].track(self.frame)
        self.player2Position, self.isPlayer2Firing = self.trackers[1].track(self.frame)

        if self.mouse_point_2 is not None:
            cv2.rectangle(self.frame, self.mouse_point_1, self.mouse_point_2, (0, 255, 0), 2)

        cv2.imshow("Camera", self.frame)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
