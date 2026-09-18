import cv2

TRAINING_VIDEO_FOLDER = "training_videos"
VIDEOS_PER_CATEGORY = 5

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


def save_video(category: str, video_index: int):
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"{TRAINING_VIDEO_FOLDER}_{category}_{video_index}",
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    print("Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        out.write(frame)

        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
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
