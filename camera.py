import cv2
from segmentation import process_contours
import numpy as np


class Camera:

    def __init__(self):

        self.h_min = 50
        self.h_max = 75
        self.s_min = 79
        self.s_max = 255
        self.v_min = 54
        self.v_max = 255

        self.direction = 0
        self.isFiring = False

        self.cap = None
        self.image_hsv = None

    def segment(self):
        if self.h_min < self.h_max:
            _, mask_h_min = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_min,
                                          maxval=1, type=cv2.THRESH_BINARY)
            _, mask_h_max = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_max,
                                          maxval=1, type=cv2.THRESH_BINARY_INV)
            mask_h = mask_h_min * mask_h_max
        else:
            _, mask_h_min = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_min,
                                          maxval=1, type=cv2.THRESH_BINARY_INV)
            _, mask_h_max = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_max,
                                          maxval=1, type=cv2.THRESH_BINARY)
            mask_h = cv2.bitwise_or(mask_h_min, mask_h_max)

        _, mask_s_min = cv2.threshold(src=self.image_hsv[:, :, 1], thresh=self.s_min,
                                      maxval=1, type=cv2.THRESH_BINARY)
        _, mask_smax = cv2.threshold(src=self.image_hsv[:, :, 1], thresh=self.s_max,
                                     maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_s = mask_s_min * mask_smax

        _, mask_v_min = cv2.threshold(src=self.image_hsv[:, :, 2], thresh=self.v_min,
                                      maxval=1, type=cv2.THRESH_BINARY)
        _, mask_v_max = cv2.threshold(src=self.image_hsv[:, :, 2], thresh=self.v_max,
                                      maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_v = mask_v_min * mask_v_max

        mask = mask_h * mask_s * mask_v

        # Applied Morphological close method to drastically reduce
        # the number of contours resulted from the mask
        # 5x5 kernel seemed to have the best results
        kernel = np.ones((5, 5), np.uint8)

        mask_close = cv2.dilate(src=mask, kernel=kernel, iterations=1)
        mask_close = cv2.erode(src=mask_close, kernel=kernel, iterations=1)

        # Comparison between the normal and closed mask
        cv2.imshow("Mask Close", mask_close * 255)
        cv2.imshow("Mask", mask * 255)

        self.direction, self.isFiring = process_contours(mask_close, self.image_hsv)
        cv2.imshow("Image", cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2BGR))

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
        self.segment()

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
