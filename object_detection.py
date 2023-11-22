from ultralytics import YOLO
import cv2
import numpy as np


class ObjectDetection:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        print("Known classes ({})".format(len(self.model.names)))
        for i in range(len(self.model.names)):
            print("{} : {}".format(i, self.model.names[i]))

    def process(self, image, object_class_1, object_class_2):
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

        results = self.model(image, verbose=False)

        image_objects = image.copy()

        position_x_1 = -1
        position_y_1 = 100

        position_x_2 = -1
        position_y_2 = 100

        objects = results[0]
        for object in objects:
            box = object.boxes.data[0]
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            confidence = box[4]
            class_id = int(box[5])

            if class_id == object_class_1 and confidence > 0.2:  # class_id == 0 and confidence > 0.8:
                cv2.rectangle(img=image_objects, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)
                text = "{}:{:.2f}".format(objects.names[class_id], confidence)
                cv2.putText(img=image_objects,
                            text=text,
                            org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 0, 0),
                            thickness=1)

                x = (pt1[0] + pt2[0]) / 2
                position_x_1 = x * 100 / image.shape[1]

                y = (pt1[1] + pt2[1]) / 2
                position_y_1 = y * 100 / image.shape[0]

            elif class_id == object_class_2 and confidence > 0.7:  # class_id == 0 and confidence > 0.8:
                cv2.rectangle(img=image_objects, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=2)
                text = "{}:{:.2f}".format(objects.names[class_id], confidence)
                cv2.putText(img=image_objects,
                            text=text,
                            org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 255),
                            thickness=1)

                x = (pt1[0] + pt2[0]) / 2
                position_x_2 = x * 100 / image.shape[1]

                y = (pt1[1] + pt2[1]) / 2
                position_y_2 = y * 100 / image.shape[0]

        image_objects = cv2.cvtColor(src=image_objects, code=cv2.COLOR_RGB2BGR)

        return image_objects, position_x_1, position_y_1 < 50, position_x_2, position_y_2 < 50
