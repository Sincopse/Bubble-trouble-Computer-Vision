import cv2
import numpy as np


def segment(image_hsv, threshold, output_as_mask):
    if threshold.h_min < threshold.h_max:
        _, mask_h_min = cv2.threshold(src=image_hsv[:, :, 0], thresh=threshold.h_min,
                                      maxval=1, type=cv2.THRESH_BINARY)
        _, mask_h_max = cv2.threshold(src=image_hsv[:, :, 0], thresh=threshold.h_max,
                                      maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_h = mask_h_min * mask_h_max
    else:
        _, mask_h_min = cv2.threshold(src=image_hsv[:, :, 0], thresh=threshold.h_min,
                                      maxval=1, type=cv2.THRESH_BINARY_INV)
        _, mask_h_max = cv2.threshold(src=image_hsv[:, :, 0], thresh=threshold.h_max,
                                      maxval=1, type=cv2.THRESH_BINARY)
        mask_h = cv2.bitwise_or(mask_h_min, mask_h_max)

    _, mask_s_min = cv2.threshold(src=image_hsv[:, :, 1], thresh=threshold.s_min,
                                  maxval=1, type=cv2.THRESH_BINARY)
    _, mask_smax = cv2.threshold(src=image_hsv[:, :, 1], thresh=threshold.s_max,
                                 maxval=1, type=cv2.THRESH_BINARY_INV)
    mask_s = mask_s_min * mask_smax

    _, mask_v_min = cv2.threshold(src=image_hsv[:, :, 2], thresh=threshold.v_min,
                                  maxval=1, type=cv2.THRESH_BINARY)
    _, mask_v_max = cv2.threshold(src=image_hsv[:, :, 2], thresh=threshold.v_max,
                                  maxval=1, type=cv2.THRESH_BINARY_INV)
    mask_v = mask_v_min * mask_v_max

    mask = mask_h * mask_s * mask_v

    # Applied Morphological open close method to reduce
    # the number of contours resulted from the mask
    # 5x5 kernel seemed to have the best results
    kernel = np.ones((5, 5), np.uint8)

    mask_close = cv2.erode(src=mask, kernel=kernel, iterations=1)
    mask_close = cv2.dilate(src=mask_close, kernel=kernel, iterations=1)

    # Comparison between the normal and closed mask
    # cv2.imshow("Mask", mask * 255)

    position, is_firing = process_contours(mask_close, image_hsv)

    if output_as_mask is False:
        return position, is_firing, cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    else:
        return position, is_firing, mask * 255


def process_contours(mask, camera_output):
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    if not contours:  # Exit if there is no contours
        return -1, False

    direction = -1
    is_firing = False

    biggest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(biggest_contour)

    if contour_area > 100:
        # draws the contour box border
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(camera_output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        m = cv2.moments(biggest_contour)

        cx = int(np.round(m['m10'] / m['m00']))  # Center x
        cy = int(np.round(m['m01'] / m['m00']))  # Center y

        cv2.circle(img=camera_output, center=(cx, cy),
                   radius=4, color=(0, 255, 255), thickness=2)

        direction = cx * 100 / mask.shape[1]

        if cy < mask.shape[0] / 2:
            cv2.rectangle(img=camera_output,
                          pt1=(10, 10),
                          pt2=(mask.shape[1] - 10, 10),
                          color=(0, 255, 255), thickness=6)
            is_firing = True

    return direction, is_firing
