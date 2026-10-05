import argparse
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

from lib import CATEGORIES, draw_text, load_recordings

DATASET_FOLDER = "datasets/yolo"
BASE_MODEL = "yolov8n.pt"
TRAINED_MODEL = "runs/detect/gesture/weights/best.pt"
VAL_SPLIT = 0.2
CONFIDENCE = 0.5


def prepare():
    """Turns the recordings made by collect.py into a YOLO dataset."""
    for split in ("train", "val"):
        os.makedirs(f"{DATASET_FOLDER}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_FOLDER}/labels/{split}", exist_ok=True)

    for recording in load_recordings():
        class_id = CATEGORIES.index(recording.category)
        # split by clip, not by frame, so val frames aren't near-duplicates of train
        split = "val" if random.random() < VAL_SPLIT else "train"
        name = os.path.splitext(os.path.basename(recording.video_path))[0].replace(
            " ", "_"
        )

        cap = cv2.VideoCapture(recording.video_path)
        for frame_index, bbox in enumerate(recording.bboxes):
            ret, frame = cap.read()
            if not ret:
                break
            if np.isnan(bbox).any():
                continue
            stem = f"{name}_{frame_index}"
            cv2.imwrite(f"{DATASET_FOLDER}/images/{split}/{stem}.jpg", frame)
            with open(f"{DATASET_FOLDER}/labels/{split}/{stem}.txt", "w") as f:
                f.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in bbox) + "\n")
        cap.release()


def write_data_yaml():
    path = f"{DATASET_FOLDER}/data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CATEGORIES))
    with open(path, "w") as f:
        f.write(
            f"path: {os.path.abspath(DATASET_FOLDER)}\n"
            "train: images/train\n"
            "val: images/val\n"
            f"names:\n{names}\n"
        )
    return path


def train():
    YOLO(BASE_MODEL).train(
        data=write_data_yaml(), epochs=50, imgsz=640, name="gesture", exist_ok=True
    )


def detect():
    model = YOLO(TRAINED_MODEL)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, conf=CONFIDENCE, verbose=False)[0]
        frame = result.plot()
        if len(result.boxes) == 0:
            draw_text(frame, "No gesture", color=(0, 0, 255))

        cv2.imshow("YOLO detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "detect"])
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "detect": detect}[args.command]()
