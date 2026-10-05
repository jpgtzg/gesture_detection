import argparse
from collections import Counter

import cv2
import numpy as np

from lib import (
    create_holistic,
    draw_landmarks,
    draw_text,
    extract_landmarks,
    load_clips,
    normalize_landmarks,
)

DATABASE_PATH = "embeddings.npz"
FRAME_STEP = 2
K = 5
MIN_SIMILARITY = 0.9


def embed(keypoints):
    """Unit-length embedding so cosine similarity is a plain dot product."""
    vector = normalize_landmarks(keypoints)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build():
    embeddings, labels = [], []
    for category, clip in load_clips():
        for keypoints in clip[::FRAME_STEP]:
            vector = embed(keypoints)
            if np.any(vector):  # skip frames where no pose was detected
                embeddings.append(vector)
                labels.append(category)

    np.savez(DATABASE_PATH, embeddings=np.array(embeddings), labels=np.array(labels))
    print(f"Saved {len(labels)} embeddings to {DATABASE_PATH}")


def classify(database_embeddings, database_labels, keypoints):
    """Returns (label, similarity) of the best match, or (None, similarity)."""
    vector = embed(keypoints)
    if not np.any(vector):
        return None, 0.0

    similarities = database_embeddings @ vector
    top = np.argsort(similarities)[-K:]
    label, _ = Counter(database_labels[top]).most_common(1)[0]
    similarity = float(similarities[top][database_labels[top] == label].mean())

    return (label if similarity >= MIN_SIMILARITY else None), similarity


def detect():
    database = np.load(DATABASE_PATH)
    database_embeddings, database_labels = database["embeddings"], database["labels"]

    holistic = create_holistic()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, results = extract_landmarks(holistic, frame)
        label, similarity = classify(database_embeddings, database_labels, keypoints)

        draw_landmarks(frame, results)
        draw_text(
            frame, label or "No gesture", color=(0, 255, 0) if label else (0, 0, 255)
        )
        draw_text(frame, f"similarity: {similarity:.2f}", y=70, scale=0.8)
        cv2.imshow("Embedding detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "detect"])
    args = parser.parse_args()
    {"build": build, "detect": detect}[args.command]()
