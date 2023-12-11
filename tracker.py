import cv2


class Tracker:

    def __init__(self):
        tracker_types = ["KCF", "CSRT"]
        tracker_type = tracker_types[1]

        if tracker_type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()

    def init_track(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def track(self, frame):
        track_ok, bbox = self.tracker.update(frame)
        if track_ok:
            x, y, w, h = bbox
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

            x = (x + (x + w)) / 2
            position_x = x * 100 / frame.shape[1]

            y = (y + (y + h)) / 2
            position_y = y * 100 / frame.shape[0]

            return position_x, position_y < 50
        else:
            cv2.putText(img=frame,
                        text="Tracking failed",
                        org=(5, 35),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=2)
            return -1, 0
