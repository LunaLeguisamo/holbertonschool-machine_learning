#!/usr/bin/env python3
"""
3-yolo.py
"""
import tensorflow as tf
import numpy as np
import os
import cv2


class Yolo:
    """YOLO v3 object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        # Carga el modelo Darknet y las clases
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [i.strip() for i in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Apply sigmoid activation function.

        Parameters:
        - x (np.ndarray): input array

        Returns:
        - np.ndarray: sigmoid output
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Transform model outputs into bounding boxes, confidences,
        and class probabilities.

        Parameters:
        - outputs (list[np.ndarray]): raw predictions for each scale
        - image_size (tuple): original image size (height, width)

        Returns:
        - boxes (list[np.ndarray]): box coordinates (x1, y1, x2, y2)
        - box_confs (list[np.ndarray]): objectness confidences
        - box_cls_probs (list[np.ndarray]): class probabilities
        """
        image_h, image_w = image_size
        boxes, box_confs, box_cls_probs = [], [], []

        for i, output in enumerate(outputs):
            gh, gw, ab, _ = output.shape

            # Obtener predicciones crudas\ + sigmoid para tx, ty y conf,
            # exp para tw, th
            t_xy = self.sigmoid(output[..., 0:2])      # tx, ty
            t_wh = output[..., 2:4]                    # tw, th (log-space)
            conf = self.sigmoid(output[..., 4:5])      # objectness
            cls = self.sigmoid(output[..., 5:])       # class probabilities

            # Construir grilla de celdas
            gx = np.arange(gw)
            gy = np.arange(gh)
            cx, cy = np.meshgrid(gx, gy)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # Calcular centros normalizados
            bx = (t_xy[..., 0] + cx) / gw
            by = (t_xy[..., 1] + cy) / gh

            # Ancho y alto relativo
            anchor_w = self.anchors[i, :, 0].reshape((1, 1, ab))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, ab))
            input_h, input_w = self.model.input.shape.as_list()[1:3]
            bw = (np.exp(t_wh[..., 0]) * anchor_w) / input_w
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / input_h

            # Convertir a coordenadas de esquina en pixeles
            x1 = (bx - bw/2) * image_w
            y1 = (by - bh/2) * image_h
            x2 = (bx + bw/2) * image_w
            y2 = (by + bh/2) * image_h
            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confs.append(conf)
            box_cls_probs.append(cls)

        return boxes, box_confs, box_cls_probs

    def filter_boxes(self, boxes, box_confs, box_cls_probs):
        """
        Filter bounding boxes by score threshold.

        Parameters:
        - boxes (list[np.ndarray]): boxes per scale
        - box_confs (list[np.ndarray]): confidences per box per scale
        - box_cls_probs (list[np.ndarray]): class probabilities
        per box per scale

        Returns:
        - filtered_boxes (np.ndarray): boxes that pass threshold
        - box_classes (np.ndarray): predicted class indices
        - box_scores (np.ndarray): corresponding scores
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for b, c, p in zip(boxes, box_confs, box_cls_probs):
            # score por clase
            scores = c * p
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            # aplicar umbral de confianza
            mask = class_scores >= self.class_t
            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        # Unir resultados de todas las escalas
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Eliminate overlapping boxes using Non-Max Suppression.

        Parameters:
        - filtered_boxes (np.ndarray): boxes from filter_boxes (N,4)
        - box_classes (np.ndarray): class indices (N,)
        - box_scores (np.ndarray): box scores (N,)

        Returns:
        - box_predictions (np.ndarray): final boxes after NMS
        - predicted_classes (np.ndarray): class for each final box
        - predicted_scores (np.ndarray): score for each final box
        """
        box_predictions = []
        predicted_classes = []
        predicted_scores = []

        # Recorrer cada clase única
        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            # Orden descendente por scores
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while len(cls_boxes) > 0:
                # Seleccionar la caja con mayor score
                best_box = cls_boxes[0]
                best_score = cls_scores[0]
                keep.append((best_box, cls, best_score))

                if len(cls_boxes) == 1:
                    break

                rest_boxes = cls_boxes[1:]
                rest_scores = cls_scores[1:]

                # Calcular coordenadas de intersección
                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                # Tamaño de intersección
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                # Área de cada caja
                best_area = (
                    best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                rest_areas = (rest_boxes[:, 2] - rest_boxes[:, 0]) * \
                             (rest_boxes[:, 3] - rest_boxes[:, 1])

                # Área de unión
                union_area = best_area + rest_areas - inter_area
                iou = inter_area / union_area

                # Filtrar cajas con IoU bajo el umbral
                mask = iou < self.nms_t
                cls_boxes = rest_boxes[mask]
                cls_scores = rest_scores[mask]

            # Agregar cajas conservadas a salidas finales
            for box, c, score in keep:
                box_predictions.append(box)
                predicted_classes.append(c)
                predicted_scores.append(score)

        return (
            np.array(box_predictions),
            np.array(predicted_classes),
            np.array(predicted_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """
        Load all images from the given folder path.

        Parameters:
        - folder_path (str): path to directory containing images

        Returns:
        - images (list[np.ndarray]): list of loaded images as arrays
        - image_paths (list[str]): list of full paths to the images
        """
        images = []
        image_paths = []

        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Construct full file path
            path = os.path.join(folder_path, filename)
            # Load image using OpenCV
            image = cv2.imread(path)
            if image is None:
                # Skip files that are not images or failed to load
                continue
            images.append(image)
            image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for YOLO model inference.

        Steps:
        1. Resize each image to the model's input size using bicubic
        interpolation.
        2. Scale pixel values to [0,1].

        Parameters:
        - images (list[np.ndarray]): original images

        Returns:
        - pimages (np.ndarray): preprocessed images of shape (ni, h, w, 3)
        - image_shapes (np.ndarray): original image shapes (ni, 2)
        """
        # Number of images
        ni = len(images)
        # Get model input dimensions (height, width)
        input_h, input_w = self.model.input.shape.as_list()[1:3]

        # Prepare arrays
        pimages = np.zeros((ni, input_h, input_w, 3), dtype=np.float32)
        image_shapes = np.zeros((ni, 2), dtype=np.int32)

        for i, img in enumerate(images):
            # Record original shape
            h, w, _ = img.shape
            image_shapes[i] = (h, w)

            # Resize image to model input size
            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

            # Normalize pixel values to [0,1]
            pimages[i] = resized / 255

        return pimages, image_shapes
