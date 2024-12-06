"""
Opencv's built in darknet api is used for 2D object detection
Result is passed into the torch net

source: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
"""

import cv2
import numpy as np
import os

class cv_Yolo:

    def __init__(self, yolo_path='default', confidence=0.5, threshold=0.3):
        # confidence is a lower bound on class confidence
        # threshold is for non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).
        # NMS iteratively removes lower scoring boxes which have an IoU greater than threshold value with another (higher scoring) box.

        yolo_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights')


        self.confidence = confidence
        self.threshold = threshold

        # coco labels
        labels_path = os.path.sep.join([yolo_path, "coco.names"])
        self.labels = open(labels_path).read().split("\n")

        # colors for labels
        np.random.seed(0)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        # rust color for boat detections
        BOAT_LABEL_INDEX = 8
        RUST_COLOR = np.array([183, 65, 14])
        self.colors[BOAT_LABEL_INDEX, :] = RUST_COLOR

        # load YOLO network configuration and weights
        cfg_path = os.path.sep.join([yolo_path, "yolov3.cfg"])
        weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def get_detections(self, image):

        (H, W) = image.shape[:2]

        # calculate the network response from input
        output_layer_name = self.net.getUnconnectedOutLayersNames()
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(output_layer_name)

        # detections is list of instances of class Detection
        detections = []
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                # detection is a 85 vector.
                # first 4 give bounding box dimensions. 5th is box confidence. Rest is class confidence scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        if len(idxs) > 0:
            for i in idxs.flatten():

                top_left = (boxes[i][0], boxes[i][1])
                bottom_right = (top_left[0] + boxes[i][2], top_left[1] + boxes[i][3])
                box_2d = [top_left, bottom_right]

                class_label = self.get_class(class_ids[i])
                if class_label == "person":
                    class_label = "pedestrian"

                detections.append(Detection(box_2d, class_label))

        return detections

    # TODO: is get_class used more than once? If not, consider to eliminate
    def get_class(self, class_label_id):
        return self.labels[class_label_id]

class Detection:
    # box_2d is list of 2 corner coordinates (tuples)
    # box_2d[0] is top left corner
    # box_2d[1] is bottom right corner
    # class_label is string
    def __init__(self, box_2d, class_label):
        self.box_2d = box_2d
        self.detected_class = class_label
