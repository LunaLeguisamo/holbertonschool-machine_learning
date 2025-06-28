#!/usr/bin/env python3
"""
7-yolo.py - YOLO v3 con predicción completa por carpeta
"""
import tensorflow as tf
import numpy as np
import os
import cv2


class Yolo:
    """YOLO v3 object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [i.strip() for i in f]
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

            t_xy = self.sigmoid(output[..., 0:2])
            t_wh = output[..., 2:4]
            conf = self.sigmoid(output[..., 4:5])
            cls = self.sigmoid(output[..., 5:])

            gx = np.arange(gw)
            gy = np.arange(gh)
            cx, cy = np.meshgrid(gx, gy)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            bx = (t_xy[..., 0] + cx) / gw
            by = (t_xy[..., 1] + cy) / gh

            anchor_w = self.anchors[i, :, 0].reshape((1, 1, ab))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, ab))
            input_h, input_w = self.model.input.shape.as_list()[1:3]
            bw = (np.exp(t_wh[..., 0]) * anchor_w) / input_w
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / input_h

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
            scores = c * p
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t
            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        box_predictions = []
        predicted_classes = []
        predicted_scores = []

        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while len(cls_boxes) > 0:
                best_box = cls_boxes[0]
                best_score = cls_scores[0]
                keep.append((best_box, cls, best_score))

                if len(cls_boxes) == 1:
                    break

                rest_boxes = cls_boxes[1:]
                rest_scores = cls_scores[1:]

                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                rest_areas = (rest_boxes[:, 2] - rest_boxes[:, 0]) * \
                             (rest_boxes[:, 3] - rest_boxes[:, 1])

                union_area = best_area + rest_areas - inter_area
                iou = inter_area / union_area

                mask = iou < self.nms_t
                cls_boxes = rest_boxes[mask]
                cls_scores = rest_scores[mask]

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
        images = []
        image_paths = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is None:
                continue
            images.append(image)
            image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        input_h, input_w = self.model.input.shape.as_list()[1:3]
        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            img = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            img = img / 255.0
            pimages.append(img)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            class_id = box_classes[i]
            score = box_scores[i]
            class_name = self.class_names[class_id]

            label = f"{class_name} {score:.2f}"
            x1, y1, x2, y2 = box

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            os.makedirs("detections", exist_ok=True)
            cv2.imwrite(f"detections/{file_name}", image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Ejecuta el pipeline completo: carga, predice y muestra.
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(pimages)
        predictions = []

        for i in range(len(images)):
            image_shape = image_shapes[i]
            # 🔧 Corrección: no expandimos dimensión de batch
            output_per_image = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                output_per_image, image_shape
            )
            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores
            )

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(images[i], boxes, box_classes, box_scores, file_name)

            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths
