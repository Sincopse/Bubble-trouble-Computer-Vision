import cv2
import numpy as np


def process_contours(mask, camera_output):
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    if not contours:  # Exit if there is no contours
        return 0, False

    direction = -1
    is_firing = False

    mask_filtered = np.zeros(mask.shape, dtype=np.uint8)

    biggest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(biggest_contour)

    if contour_area > 100:
        # draw the biggest contour rectangle border
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(camera_output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.drawContours(image=mask_filtered, contours=biggest_contour,
                         contourIdx=-1, color=(1, 1, 1), thickness=-1)
        m = cv2.moments(biggest_contour)

        cx = int(np.round(m['m10'] / m['m00']))  # Center x
        cy = int(np.round(m['m01'] / m['m00']))  # Center y
        cv2.circle(img=camera_output, center=(cx, cy),
                   radius=4, color=(0, 255, 255), thickness=2)

        direction = cx * 100 / mask.shape[1]

        if cy < mask.shape[0] / 2:
            cv2.rectangle(img=mask_filtered,
                          pt1=(10, 10),
                          pt2=(mask.shape[1] - 10, 10),
                          color=(1, 1, 1), thickness=6)
            is_firing = True

    cv2.imshow("Mask Filtered", mask_filtered * 255)

    return direction, is_firing
