#!/usr/bin/env python3
"""
3-yolo.py
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """YOLO v3 object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        # Carga el modelo Darknet y las clases
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [l.strip() for l in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        image_h, image_w = image_size
        boxes, box_confs, box_cls_probs = [], [], []

        for i, output in enumerate(outputs):
            gh, gw, ab, _ = output.shape

            # 1) extraer t_xy, t_wh, conf, class_probs
            t_xy = self.sigmoid(output[..., 0:2])
            t_wh = output[..., 2:4]
            conf = self.sigmoid(output[..., 4:5])
            cls = self.sigmoid(output[..., 5:])

            # 2) grid
            gx = np.arange(gw)
            gy = np.arange(gh)
            cx, cy = np.meshgrid(gx, gy)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # 3) bx, by
            bx = (t_xy[..., 0] + cx) / gw
            by = (t_xy[..., 1] + cy) / gh

            # 4) bw, bh (relativo a la imagen)
            anchor_w = self.anchors[i, :, 0].reshape((1, 1, ab))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, ab))
            input_h, input_w = self.model.input.shape.as_list()[1:3]
            bw = (np.exp(t_wh[..., 0]) * anchor_w) / input_w
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / input_h

            # 5) corner coordinates (x1,y1,x2,y2) en pixeles
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
        filtered_boxes, box_classes, box_scores = [], [], []

        for b, c, p in zip(boxes, box_confs, box_cls_probs):
            # score por clase
            scores = c * p  # (gh, gw, ab, classes)
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            # aplica umbral
            mask = class_scores >= self.class_t
            # filtra
            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        # concatena todas las escalas
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes    = np.concatenate(box_classes,    axis=0)
        box_scores     = np.concatenate(box_scores,     axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        box_predictions = []
        predicted_classes = []
        predicted_scores = []

        # trabajar por cada clase
        for cls in np.unique(box_classes):
            # extraer solo las cajas de esa clase
            idxs = np.where(box_classes == cls)
            cls_boxes  = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            # ordenar por score descendente
            order = np.argsort(cls_scores)[::-1]
            cls_boxes  = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while len(cls_boxes) > 0:
                # cogemos la caja de mayor score
                best_box = cls_boxes[0]
                best_score = cls_scores[0]
                keep.append((best_box, cls, best_score))

                if len(cls_boxes) == 1:
                    break

                # resto de cajas a comparar
                rest_boxes  = cls_boxes[1:]
                rest_scores = cls_scores[1:]

                # calcular intersección
                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                # áreas individuales
                best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                rest_areas = (rest_boxes[:, 2] - rest_boxes[:, 0]) * \
                             (rest_boxes[:, 3] - rest_boxes[:, 1])

                # unión = suma - intersección
                union_area = best_area + rest_areas - inter_area
                iou = inter_area / union_area

                # quedarnos solo con IoU < umbral
                mask = iou < self.nms_t
                cls_boxes  = rest_boxes[mask]
                cls_scores = rest_scores[mask]

            # acumular resultados finales
            for box, c, score in keep:
                box_predictions.append(box)
                predicted_classes.append(c)
                predicted_scores.append(score)

        return (
            np.array(box_predictions),
            np.array(predicted_classes),
            np.array(predicted_scores)
        )
