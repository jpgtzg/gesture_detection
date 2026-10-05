import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .config import LEGACY_LANDMARKS_FOLDER, RECORDINGS_FOLDER


@dataclass
class Recording:
    category: str
    landmarks: np.ndarray  # (frames, 225)
    bboxes: np.ndarray  # (frames, 4) normalized x_center, y_center, w, h; NaN if none
    fps: float
    video_path: str


def load_recordings(folder: str = RECORDINGS_FOLDER):
    """Loads everything saved by collect.py."""
    recordings = []
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        data = np.load(path)
        recordings.append(
            Recording(
                category=str(data["category"]),
                landmarks=data["landmarks"],
                bboxes=data["bboxes"],
                fps=float(data["fps"]),
                video_path=os.path.splitext(path)[0] + ".mp4",
            )
        )
    return recordings


def load_clips():
    """Landmark sequences as a list of (category, array of shape (frames, 225)).

    Includes the old landmark-only clips so earlier recordings aren't wasted.
    """
    clips = [(r.category, r.landmarks) for r in load_recordings()]
    for path in sorted(glob.glob(os.path.join(LEGACY_LANDMARKS_FOLDER, "*.npy"))):
        # files are named <category>_<index>.npy
        category = re.sub(r"_\d+$", "", os.path.splitext(os.path.basename(path))[0])
        clips.append((category, np.load(path)))
    return clips
