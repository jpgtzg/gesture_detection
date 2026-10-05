import os
import time

import cv2
import numpy as np

from .config import COUNTDOWN_SECONDS, RECORDING_DURATION_SECONDS, RECORDINGS_FOLDER
from .landmarks import draw_landmarks, extract_landmarks, landmarks_bbox


def draw_text(frame, text, y=30, scale=1, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def wait_for_start(cap, category: str):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_text(frame, category)
        draw_text(frame, "Press SPACE to start recording", y=70, scale=0.8)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord(" "):
            break


def countdown(cap, category: str):
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        remaining = COUNTDOWN_SECONDS - (time.time() - start_time)
        if remaining <= 0:
            break

        draw_text(frame, category)
        draw_text(
            frame, f"Starting in {remaining:.1f}s", y=70, scale=0.8, color=(0, 0, 255)
        )
        cv2.imshow("Recording", frame)
        cv2.waitKey(1)


def record_clip(holistic, category: str):
    """Records one clip from the webcam.

    Returns (landmark vectors, raw frames, measured fps).
    """
    cap = cv2.VideoCapture(0)

    wait_for_start(cap, category)
    countdown(cap, category)

    keypoints_sequence, frames = [], []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, results = extract_landmarks(holistic, frame)
        keypoints_sequence.append(keypoints)
        frames.append(frame.copy())  # copy: the drawing below modifies frame

        draw_landmarks(frame, results)

        elapsed = time.time() - start_time
        remaining = max(0.0, RECORDING_DURATION_SECONDS - elapsed)
        draw_text(frame, category)
        draw_text(frame, f"{remaining:.1f}s", y=70)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or elapsed >= RECORDING_DURATION_SECONDS:
            break

    cap.release()
    cv2.destroyAllWindows()

    fps = len(frames) / max(time.time() - start_time, 1e-6)
    return keypoints_sequence, frames, fps


def save_recording(category: str, index: int, keypoints_sequence, frames, fps: float):
    """Saves <category>_<index>.mp4 (raw video) and .npz (landmarks, bboxes, fps)."""
    os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
    base = f"{RECORDINGS_FOLDER}/{category}_{index}"

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        f"{base}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

    # one (x_center, y_center, w, h) per frame, NaN where nothing was detected
    bboxes = np.array(
        [
            bbox if bbox is not None else (np.nan,) * 4
            for bbox in map(landmarks_bbox, keypoints_sequence)
        ]
    )
    np.savez(
        f"{base}.npz",
        landmarks=np.array(keypoints_sequence),
        bboxes=bboxes,
        fps=fps,
        category=category,
    )
