import cv2
import numpy as np


def process_contours(mask):
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    if not contours:  # Exit if there is no contours
        return 0, False

    direction = 0
    is_firing = False

    mask_filtered = np.zeros(mask.shape, dtype=np.uint8)

    for i in range(len(contours)):
        contour = contours[i]
        contour_area = cv2.contourArea(contour)
        if contour_area > 100:
            cv2.drawContours(image=mask_filtered, contours=contours,
                             contourIdx=i, color=(1, 1, 1), thickness=-1)
            m = cv2.moments(contour)
            cx = int(np.round(m['m10'] / m['m00']))  # Center x
            cy = int(np.round(m['m01'] / m['m00']))  # Center y
            perimeter = cv2.arcLength(curve=contour, closed=True)
            if cx > (2 / 3) * mask.shape[1]:
                cv2.rectangle(img=mask_filtered,
                              pt1=(mask.shape[1] - 10, 0),
                              pt2=(mask.shape[1] - 10, mask.shape[0]),
                              color=(1, 1, 1), thickness=6)
                cv2.imshow("Mask Filtered", mask_filtered * 255)
                direction = 1

            elif cx < (1 / 3) * mask.shape[1]:
                cv2.rectangle(img=mask_filtered,
                              pt1=(10, 0),
                              pt2=(10, mask.shape[0]),
                              color=(1, 1, 1), thickness=6)
                cv2.imshow("Mask Filtered", mask_filtered * 255)
                direction = -1

            else:
                cv2.imshow("Mask Filtered", mask_filtered * 255)
                direction = 0

            if cy < mask.shape[0] / 2:
                cv2.rectangle(img=mask_filtered,
                              pt1=(10, 10),
                              pt2=(mask.shape[1] - 10, 10),
                              color=(1, 1, 1), thickness=6)
                cv2.imshow("Mask Filtered", mask_filtered * 255)
                is_firing = True

            else:
                is_firing = False

    cv2.imshow("Mask Filtered", mask_filtered * 255)

    return direction, is_firing


def process_contours_new(mask):
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    if not contours:  # Exit if there is no contours
        return 0, False

    direction = 0
    is_firing = False

    mask_filtered = np.zeros(mask.shape, dtype=np.uint8)

    biggest_contour = contours[0]
    biggest_contour_area = 0

    # Select the biggest contour
    for i in range(len(contours)):
        biggest_contour_area = cv2.contourArea(biggest_contour)
        contour_area         = cv2.contourArea(contours[i])

        if contour_area > biggest_contour_area:  # Need to have at least 100 pixels area
            biggest_contour = contours[i]

    if biggest_contour is not None and biggest_contour_area > 0:
        cv2.drawContours(image=mask_filtered, contours=contours,
                         contourIdx=0, color=(1, 1, 1), thickness=-1)
        m = cv2.moments(biggest_contour)
        cx = int(np.round(m['m10'] / m['m00']))  # Center x
        cy = int(np.round(m['m01'] / m['m00']))  # Center y

        cv2.circle(img=mask_filtered, center=(cx, cy),
                   radius=6, color=(0, 0, 0), thickness=2)

        if cx > (2 / 3) * mask.shape[1]:
            cv2.rectangle(img=mask_filtered,
                          pt1=(mask.shape[1] - 10, 0),
                          pt2=(mask.shape[1] - 10, mask.shape[0]),
                          color=(1, 1, 1), thickness=6)
            direction = 1

        elif cx < (1 / 3) * mask.shape[1]:
            cv2.rectangle(img=mask_filtered,
                          pt1=(10, 0),
                          pt2=(10, mask.shape[0]),
                          color=(1, 1, 1), thickness=6)
            direction = -1

        if cy < mask.shape[0] / 2:
            cv2.rectangle(img=mask_filtered,
                          pt1=(10, 10),
                          pt2=(mask.shape[1] - 10, 10),
                          color=(1, 1, 1), thickness=6)
            is_firing = True

    cv2.imshow("Mask Filtered", mask_filtered * 255)

    return direction, is_firing
