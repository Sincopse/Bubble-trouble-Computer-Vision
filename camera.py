import cv2
from segmentation import segment


class Camera:

    def __init__(self):

        self.h_min = 50
        self.h_max = 75
        self.s_min = 79
        self.s_max = 255
        self.v_min = 54
        self.v_max = 255

        self.direction = 0
        self.is_firing = False

        self.cap = None
        self.image_hsv = None

    # <editor-fold desc="TrackBarUpdaters">

    def on_change_h_min(self, val):
        self.h_min = val

    def on_change_h_max(self, val):
        self.h_max = val

    def on_change_s_min(self, val):
        self.s_min = val

    def on_change_s_max(self, val):
        self.s_max = val

    def on_change_v_min(self, val):
        self.v_min = val

    def on_change_v_max(self, val):
        self.v_max = val

    # </editor-fold>

    def open_camera(self):
        self.cap = cv2.VideoCapture()
        if not self.cap.isOpened():
            self.cap.open(0)
        ret, image = self.cap.read()
        self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cv2.namedWindow("Image")
        cv2.createTrackbar("H_min", "Image", self.h_min, 180, self.on_change_h_min)
        cv2.createTrackbar("H_max", "Image", self.h_max, 180, self.on_change_h_max)
        cv2.createTrackbar("S_min", "Image", self.s_min, 255, self.on_change_s_min)
        cv2.createTrackbar("S_max", "Image", self.s_max, 255, self.on_change_s_max)
        cv2.createTrackbar("V_min", "Image", self.v_min, 255, self.on_change_v_min)
        cv2.createTrackbar("V_max", "Image", self.v_max, 255, self.on_change_v_max)

    def update_camera(self):
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()
        image = image[:, ::-1, :]
        cv2.imshow("Image", image)
        self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.direction, self.is_firing = segment(self.image_hsv,
                                       self.h_min, self.h_max,
                                       self.s_min, self.s_max,
                                       self.v_min, self.v_max)

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
