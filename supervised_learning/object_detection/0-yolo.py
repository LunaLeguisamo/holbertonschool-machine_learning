#!/usr/bin/env python3
"""
Class Yolo that uses the Yolo v3 algorithm to perform object detection
"""

from tensorflow import keras as K
import numpy as np


class Yolo:
    """YOLO v3 object detection model"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO with config and model"""
        self.model = K.models.load_model(model_path)

        # Cargar nombres de las clases
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
