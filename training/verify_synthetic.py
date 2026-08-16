import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import numpy as np
import os

for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

model_path = r"E:\SCRIPTS\Games_Related\poke.AI\training\map_detector_synthetic.h5"
data_dir = r"E:\SCRIPTS\Games_Related\poke.AI\training\data"

# same load path as ai/standalone_backend.py
model = models.load_model(model_path, backbone_name="resnet50")
print("Model loaded OK")

# class name -> id (same mapping as classes.csv used in training)
class_ids = {}
with open(os.path.join(data_dir, "classes.csv")) as f:
    for line in f:
        name, idx = line.strip().split(",")
        class_ids[name] = int(idx)
id_names = {v: k for k, v in class_ids.items()}

# ground truth from val labels
gt = {}
with open(os.path.join(data_dir, "val_labels.csv")) as f:
    for line in f:
        parts = line.strip().split(",")
        name = os.path.join(data_dir, *parts[0].split("/"))
        gt.setdefault(name, []).append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), class_ids[parts[5]]))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)

total_tp = 0
total_gt = 0
for path, labels in gt.items():
    frame = np.array(read_image_bgr(path), copy=True)
    image = preprocess_image(frame)
    image, scale = resize_image(image, min_side=320)
    boxes, scores, labels_out = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    dets = [(b.astype(int).tolist(), float(s), int(l)) for b, s, l in zip(boxes[0], scores[0], labels_out[0]) if s >= 0.5]
    matched = set()
    tp = 0
    for l in labels:
        total_gt += 1
        best = max(((iou(l, d[0]), i) for i, d in enumerate(dets) if d[2] == l[4] and i not in matched), default=(0.0, -1))
        if best[0] >= 0.5:
            matched.add(best[1])
            tp += 1
    total_tp += tp
    print("{}: gt={} dets={} correct={}".format(os.path.basename(path), len(labels), len(dets), tp))

print("TOTAL: {}/{} ground-truth boxes detected correctly (IoU>=0.5, same class)".format(total_tp, total_gt))
