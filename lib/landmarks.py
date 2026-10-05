import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

POSE_POINTS = 33
HAND_POINTS = 21
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def create_holistic():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _landmarks_to_array(landmark_list, num_points):
    if landmark_list is None:
        return np.zeros(num_points * 3)  # x, y, z per point, zeroed if not detected
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]).flatten()


def extract_landmarks(holistic, frame):
    """Returns (flat vector of pose + left hand + right hand landmarks, mediapipe results)."""
    # MediaPipe expects RGB, OpenCV gives BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    pose = _landmarks_to_array(results.pose_landmarks, POSE_POINTS)
    left_hand = _landmarks_to_array(results.left_hand_landmarks, HAND_POINTS)
    right_hand = _landmarks_to_array(results.right_hand_landmarks, HAND_POINTS)

    return np.concatenate([pose, left_hand, right_hand]), results


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def normalize_landmarks(keypoints):
    """Makes a landmark vector invariant to position and distance from the camera.

    Points are centered on the middle of the shoulders and divided by the shoulder
    width. Points that weren't detected (all zeros) stay zero. Returns a zero vector
    if the pose itself wasn't detected.
    """
    points = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    detected = np.any(points != 0, axis=1)

    left, right = points[LEFT_SHOULDER], points[RIGHT_SHOULDER]
    scale = np.linalg.norm(left[:2] - right[:2])
    if not detected[LEFT_SHOULDER] or not detected[RIGHT_SHOULDER] or scale < 1e-6:
        return np.zeros(points.size)

    center = (left + right) / 2
    normalized = (points - center) / scale
    normalized[~detected] = 0
    return normalized.flatten()


def landmarks_bbox(keypoints, margin=0.05):
    """Normalized (x_center, y_center, width, height) around every detected landmark.

    Returns None if nothing was detected.
    """
    points = np.asarray(keypoints).reshape(-1, 3)
    points = points[np.any(points != 0, axis=1)][:, :2]
    if len(points) == 0:
        return None

    x_min, y_min = np.clip(points.min(axis=0) - margin, 0, 1)
    x_max, y_max = np.clip(points.max(axis=0) + margin, 0, 1)
    return (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        x_max - x_min,
        y_max - y_min,
    )
