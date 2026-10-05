import os
import time

import cv2
import mediapipe as mp
import numpy as np

TRAINING_VIDEO_FOLDER = "training_videos"
VIDEOS_PER_CATEGORY = 5
RECORDING_DURATION_SECONDS = 5
COUNTDOWN_SECONDS = 3

CATEGORIES = [
    # "release_arm",
    "shake_hand",
    "high_five",
    "hug",
    "high_wave",
    "clap",
    # "face_wave",
    "left_kiss",
    "heart",
    "right heart",
    "hands_up",
    # "x-ray",
    "right_hand_up",
    # "reject",
    "right_kiss",
    "two_hand_kiss",
]


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_landmarks(frame):
    # MediaPipe expects RGB, OpenCV gives BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    def landmarks_to_array(landmark_list, num_points):
        if landmark_list is None:
            return np.zeros(num_points * 3)  # x, y, z per point, zeroed if not detected
        return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]).flatten()

    pose = landmarks_to_array(results.pose_landmarks, 33)
    left_hand = landmarks_to_array(results.left_hand_landmarks, 21)
    right_hand = landmarks_to_array(results.right_hand_landmarks, 21)

    return np.concatenate([pose, left_hand, right_hand]), results


def wait_for_start(cap, category: str):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            category,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press SPACE to start recording",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord(" "):
            break


def countdown(cap, category: str):
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        remaining = COUNTDOWN_SECONDS - elapsed
        if remaining <= 0:
            break

        cv2.putText(
            frame,
            category,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Starting in {remaining:.1f}s",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Recording", frame)
        cv2.waitKey(1)


def save_video(category: str, video_index: int):
    cap = cv2.VideoCapture(0)

    wait_for_start(cap, category)
    countdown(cap, category)

    frame_sequence = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, results = extract_landmarks(frame)
        frame_sequence.append(keypoints)

        # optional: draw for visual feedback while recording
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

        elapsed = time.time() - start_time
        remaining = max(0.0, RECORDING_DURATION_SECONDS - elapsed)

        cv2.putText(
            frame,
            category,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{remaining:.1f}s",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or elapsed >= RECORDING_DURATION_SECONDS:
            break

    cap.release()
    cv2.destroyAllWindows()

    # save the whole clip's keypoints as one .npy file, shape (num_frames, num_features)
    np.save(
        f"{TRAINING_VIDEO_FOLDER}/{category}_{video_index}.npy",
        np.array(frame_sequence),
    )


if __name__ == "__main__":
    os.makedirs(TRAINING_VIDEO_FOLDER, exist_ok=True)
    print(
        f"There are {len(CATEGORIES)} categories available for training. For each category, we'll record {VIDEOS_PER_CATEGORY} videos"
    )
    for category in CATEGORIES:
        print(f"Recording videos for category: {category}")
        for i in range(VIDEOS_PER_CATEGORY):
            print(
                f"Recording video {i + 1}/{VIDEOS_PER_CATEGORY} for category: {category}"
            )
            save_video(category=category, video_index=i)
