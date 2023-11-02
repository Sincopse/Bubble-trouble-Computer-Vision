import cv2
from segmentation import segment


class Thresholds:

    def __init__(self):
        self.h_min = 60
        self.h_max = 96
        self.s_min = 79
        self.s_max = 255
        self.v_min = 54
        self.v_max = 255

        self.position = 0
        self.is_firing = False

    # <editor-fold desc="TrackBarUpdaters">

    def on_change_h_min(self, val): self.h_min = val

    def on_change_h_max(self, val): self.h_max = val

    def on_change_s_min(self, val): self.s_min = val

    def on_change_s_max(self, val): self.s_max = val

    def on_change_v_min(self, val): self.v_min = val

    def on_change_v_max(self, val): self.v_max = val

    # </editor-fold>

    def open_camera(self, number):
        window_name = "Image " + str(number)
        cv2.namedWindow(window_name)
        cv2.createTrackbar("H_min", window_name, self.h_min, 180, self.on_change_h_min)
        cv2.createTrackbar("H_max", window_name, self.h_max, 180, self.on_change_h_max)
        cv2.createTrackbar("S_min", window_name, self.s_min, 255, self.on_change_s_min)
        cv2.createTrackbar("S_max", window_name, self.s_max, 255, self.on_change_s_max)
        cv2.createTrackbar("V_min", window_name, self.v_min, 255, self.on_change_v_min)
        cv2.createTrackbar("V_max", window_name, self.v_max, 255, self.on_change_v_max)

    def get_player_action_with_segmentation(self, image_hsv, output_as_mask):
        position, self.is_firing, camera_image = segment(image_hsv, self, output_as_mask)
        if position != -1:
            self.position = position
        return camera_image
