#!/usr/bin/env python3
"""
0-yolo.py
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """
    Yolo class uses the YOLO v3 algorithm to perform object detection.

    Attributes:
        model (keras.Model): The Darknet Keras model loaded from model_path.
        class_names (list): List of class names for the model.
        class_t (float): Box score threshold for the initial filtering step.
        nms_t (float): Intersection over Union (IoU) threshold for non-max
            suppression.
        anchors (np.ndarray): Anchor boxes used by the model.
            Shape is (outputs, anchor_boxes, 2), where:
            - outputs: number of output layers of the Darknet model
            - anchor_boxes: number of anchor boxes per output
            - 2: width and height of each anchor box.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object detector.

        Parameters:
        - model_path (str): path to the Darknet Keras model.
        - classes_path (str): path to the file containing class
            names used by the model.
        - class_t (float): box score threshold for the initial filtering step.
        - nms_t (float): IOU threshold for non-max suppression.
        - anchors (np.ndarray): array of shape (outputs, anchor_boxes, 2)
            with anchor box dimensions.
            outputs: number of output predictions made by the model
            anchor_boxes: number of anchor boxes per prediction
            2: [anchor_box_width, anchor_box_height]
        """

        self.model = tf.keras.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Apply the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the YOLO model.

        Parameters:
        - outputs: list of arrays (one for each scale of the model)
        - image_size: actual image size [height, width]

        Returns:
        - boxes: coordinates (x1, y1, x2, y2) rescaled to the original image
        - box_confidences: object confidence for each box
        - box_class_probs: class probabilities for each box
        """
        image_h, image_w = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]

            # Separamos los datos del output
            t_xy = self.sigmoid(output[..., 0:2])      # tx, ty
            t_wh = output[..., 2:4]                    # tw, th
            # confianza de que hay un objeto
            objectness = self.sigmoid(output[..., 4:5])
            # probabilidades por clase
            class_probs = self.sigmoid(output[..., 5:])

            # Creamos coordenadas (cx, cy) de la grilla
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # Sumamos la posición de la celda para calcular bx, by
            bx = (t_xy[..., 0] + cx) / grid_w
            by = (t_xy[..., 1] + cy) / grid_h

            # Ancho y alto usando anchor boxes
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            bw = (np.exp(t_wh[..., 0]) * anchor_w) / self.model.input.shape[1]
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / self.model.input.shape[2]

            # Convertimos a (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            # Apilamos las coordenadas en el último eje
            box = np.stack([x1, y1, x2, y2], axis=-1)

            # Guardamos los resultados
            boxes.append(box)
            box_confidences.append(objectness)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes based on objectness score and class
        probabilities.

        Parameters:
        - boxes: list of (grid_h, grid_w, anchor_boxes, 4)
        - box_confidences: list of (grid_h, grid_w, anchor_boxes, 1)
        - box_class_probs: list of (grid_h, grid_w, anchor_boxes, classes)

        Returns:
        - filtered_boxes: (?, 4)
        - box_classes: (?,)
        - box_scores: (?)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # 1. Score por clase = confianzas * probabilidad por clase
            box_score = box_confidences[i] * box_class_probs[i]

            # 2. Clase con mayor score y el score asociado
            classes = np.argmax(box_score, axis=-1)  # (h, w, anchors)
            scores = np.max(box_score, axis=-1)      # (h, w, anchors)

            # 3. Filtro por score mínimo
            mask = scores >= self.class_t

            # 4. Aplicamos la máscara (flatten y filtramos)
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(scores[mask])

        # 5. Concatenamos los resultados de todas las escalas
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        
        index = np.argsort(box_scores)[::-1]
        sorted_scores = box_scores[index]
        sorted_boxes = filtered_boxes[index]
        sorted_classes = box_classes[index]

        max_score_box = box_scores[0]
        