"""Records gesture clips for every approach.

Each clip is saved as recordings/<category>_<index>.mp4 (raw video) plus
recordings/<category>_<index>.npz (MediaPipe landmarks, bounding boxes, fps).
Approach 1 uses the video + bounding boxes, approach 2 the landmarks.

    uv run collect.py                                  # all categories
    uv run collect.py -c clap hug -n 3                 # 3 more clips of clap and hug
"""

import argparse
import glob

from lib import (
    CATEGORIES,
    RECORDINGS_FOLDER,
    create_holistic,
    record_clip,
    save_recording,
)
from lib.config import VIDEOS_PER_CATEGORY


def next_index(category: str) -> int:
    existing = glob.glob(f"{RECORDINGS_FOLDER}/{category}_*.npz")
    return len(existing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES
    )
    parser.add_argument("-n", "--videos", type=int, default=VIDEOS_PER_CATEGORY)
    args = parser.parse_args()

    holistic = create_holistic()
    print(
        f"Recording {args.videos} videos for each of {len(args.categories)} categories"
    )
    for category in args.categories:
        print(f"Recording videos for category: {category}")
        for i in range(args.videos):
            print(f"Recording video {i + 1}/{args.videos} for category: {category}")
            keypoints, frames, fps = record_clip(holistic, category)
            if frames:
                save_recording(category, next_index(category), keypoints, frames, fps)
