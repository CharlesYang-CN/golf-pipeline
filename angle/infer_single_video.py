import json
import numpy as np
import cv2
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

VIDEO_PATH = "/workspace/angle/24.mp4"
MODEL_PATH = "/workspace/angle/pose_landmarker.task"
OUTPUT_PATH = "/workspace/angle/keypoints.json"


def mean_points(points, ids):
    return np.mean([points[i][:2] for i in ids], axis=0)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def estimate_club(points, handedness="right"):
    left_hand = mean_points(points, [15, 17, 19, 21])
    right_hand = mean_points(points, [16, 18, 20, 22])
    grip = (left_hand + right_hand) / 2

    if handedness == "right":
        shaft_dir = normalize(right_hand - left_hand)
    else:
        shaft_dir = normalize(left_hand - right_hand)

    if shaft_dir is None:
        return None

    ys = [p[1] for p in points]
    body_height = max(ys) - min(ys)
    club_length = 0.60 * body_height

    club_head = grip + club_length * shaft_dir
    butt_end = grip - 0.10 * club_length * shaft_dir

    return {
        "grip": grip.tolist(),
        "butt_end": butt_end.tolist(),
        "club_head": club_head.tolist(),
        "shaft_line": [butt_end.tolist(), club_head.tolist()],
    }


cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)

all_frames = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_idx)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        pts = np.array([[lm.x, lm.y] for lm in landmarks])
        frame_data = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in landmarks
        ]
        club = estimate_club(pts)
    else:
        frame_data = []
        club = None

    all_frames.append({"frame": frame_idx, "keypoints": frame_data, "club": club})
    frame_idx += 1

cap.release()
landmarker.close()

output = {
    "fps": fps,
    "total_frames": frame_idx,
    "num_landmarks": 33,
    "landmark_names": [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ],
    "data": all_frames,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Done: {frame_idx} frames -> {OUTPUT_PATH}")
detected = sum(1 for f in all_frames if f["keypoints"])
print(f"Frames with pose detected: {detected}/{frame_idx}")
