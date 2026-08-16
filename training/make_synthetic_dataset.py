"""Generate a small synthetic overworld-tile dataset to smoke-test the
keras-retinanet training -> conversion -> inference pipeline on the GPU.

Each image is a dark "overworld" background with a handful of solid colored
rectangles standing in for the map tiles. The color uniquely identifies the
class, so the model learns the tile classes quickly and we can verify the
whole pipeline works end to end (real gameplay footage + labelImg labels can
be dropped into the same CSV format later - see README.md in this folder).

Outputs (all under ./data relative to this file):
    classes.csv            class name -> id mapping (same 8 classes as the
                           original poke-AI overworld_detection labels.csv)
    train_labels.csv       keras-retinanet CSV annotations for the train set
    val_labels.csv         keras-retinanet CSV annotations for the val set
    train/*.png            generated training images
    val/*.png              generated validation images

Usage:
    uv run python training/make_synthetic_dataset.py
"""

import os
import random

import cv2
import numpy as np

IMG_SIZE = 320
N_TRAIN = 40
N_VAL = 12
SEED = 42

# class name -> BGR color used to draw the tile on the synthetic image
CLASS_COLORS = {
    "pokecenter": (255, 144, 30),
    "pokemart": (0, 140, 255),
    "npc": (50, 50, 220),
    "house": (43, 90, 139),
    "gym": (240, 32, 160),
    "exit": (255, 255, 0),
    "wall": (128, 128, 128),
    "grass": (0, 160, 0),
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def make_image(rng, out_path, labels_out):
    img = np.full((IMG_SIZE, IMG_SIZE, 3), (30, 30, 30), dtype=np.uint8)
    boxes = []
    n_boxes = rng.randint(3, 8)
    attempts = 0
    while len(boxes) < n_boxes and attempts < 200:
        attempts += 1
        w = rng.randint(24, 56)
        h = rng.randint(24, 56)
        x1 = rng.randint(8, IMG_SIZE - w - 8)
        y1 = rng.randint(8, IMG_SIZE - h - 8)
        x2 = x1 + w
        y2 = y1 + h
        if any(not (x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1) for bx1, by1, bx2, by2, _ in boxes):
            continue
        cls = rng.choice(list(CLASS_COLORS.keys()))
        boxes.append((x1, y1, x2, y2, cls))
        cv2.rectangle(img, (x1, y1), (x2, y2), CLASS_COLORS[cls], thickness=-1)
    cv2.imwrite(out_path, img)
    for x1, y1, x2, y2, cls in boxes:
        labels_out.append("{},{},{},{},{},{}".format(
            os.path.relpath(out_path, DATA_DIR).replace("\\", "/"),
            x1, y1, x2, y2, cls))


def main():
    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val"), exist_ok=True)

    with open(os.path.join(DATA_DIR, "classes.csv"), "w") as f:
        for i, cls in enumerate(CLASS_COLORS):
            f.write("{},{}\n".format(cls, i))

    rng = random.Random(SEED)
    for split, count, labels_file in (("train", N_TRAIN, "train_labels.csv"),
                                      ("val", N_VAL, "val_labels.csv")):
        labels_out = []
        for i in range(count):
            out_path = os.path.join(DATA_DIR, split, "img_{:03d}.png".format(i))
            make_image(rng, out_path, labels_out)
        with open(os.path.join(DATA_DIR, labels_file), "w") as f:
            f.write("\n".join(labels_out) + "\n")
        print("{}: {} images, {} boxes -> {}".format(split, count, len(labels_out), labels_file))


if __name__ == "__main__":
    main()
