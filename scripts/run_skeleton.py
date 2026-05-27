"""Wrapper for MediaPipe pose keypoint extraction + 2D skeleton visualization.

Usage: conda run -n env_angle python scripts/run_skeleton.py <video_path> <output_dir>
Outputs:
  - output_dir/keypoints.json   (33 landmarks per frame)
  - output_dir/robot_golf.mp4   (2D skeleton animation video)
"""
import argparse
import json
import os
import sys
from collections import deque

import cv2
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

MODEL_PATH = '/app/angle/pose_landmarker.task'


def extract_keypoints(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
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
            club = _estimate_club(pts)
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
        "landmark_names": LANDMARK_NAMES,
        "data": all_frames,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Keypoints extracted: {frame_idx} frames -> {output_path}")
    return output_path, fps


def render_skeleton_video(kps_json_path, output_video, fps=None):
    with open(kps_json_path) as f:
        data = json.load(f)
    frames = data['data']
    if fps is None:
        fps = data.get('fps', 30)

    W, H = 1280, 720

    skeleton_edges = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    ]
    face_and_feet = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (1, 4), (7, 8), (9, 10),
        (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32),
    ]

    trail = deque(maxlen=18)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    prev_club_head = None

    for fi, frame_data in enumerate(frames):
        canvas = _make_background(W, H)
        pts = frame_data.get('keypoints', [])
        if not pts:
            out.write(canvas)
            continue

        kps = np.array([[p['x'] * W, p['y'] * H] for p in pts])

        for (i, j) in face_and_feet + skeleton_edges:
            if kps[i][0] > 0 and kps[j][0] > 0:
                _draw_glow_line(canvas, tuple(kps[i].astype(int)),
                               tuple(kps[j].astype(int)), (100, 200, 255), 3)

        for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if kps[i][0] > 0:
                cv2.circle(canvas, tuple(kps[i].astype(int)), 5, (0, 255, 255), -1)

        club = frame_data.get('club')
        if club and club.get('club_head'):
            club_head = np.array(club['club_head']) * [W, H]
            trail.append(tuple(club_head.astype(int)))
            if prev_club_head is not None:
                pt1 = tuple(prev_club_head.astype(int))
                pt2 = tuple(club_head.astype(int))
                cv2.line(canvas, pt1, pt2, (0, 255, 100), 2, cv2.LINE_AA)
            prev_club_head = club_head

        if len(trail) > 1:
            pts_arr = np.array(trail, dtype=np.int32)
            for k in range(1, len(pts_arr)):
                cv2.line(canvas, tuple(pts_arr[k - 1]), tuple(pts_arr[k]),
                        (0, int(255 * k / len(trail)), 100), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Frame: {fi}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(canvas)

    out.release()
    print(f"Skeleton video rendered: {output_video}")


def _make_background(W, H):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        val = int(20 + 20 * y / H)
        canvas[y, :] = (val, val // 2, val // 2 + 10)
    return canvas


def _draw_glow_line(img, pt1, pt2, color, thickness):
    overlay = img.copy()
    for t in [thickness + 4, thickness + 2, thickness]:
        alpha = 0.3 - 0.08 * (thickness - t)
        cv2.line(overlay, pt1, pt2, color, t, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def _mean_points(points, ids):
    return np.mean([points[i][:2] for i in ids], axis=0)


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def _estimate_club(points, handedness="right"):
    left_hand = _mean_points(points, [15, 17, 19, 21])
    right_hand = _mean_points(points, [16, 18, 20, 22])
    grip = (left_hand + right_hand) / 2
    if handedness == "right":
        shaft_dir = _normalize(right_hand - left_hand)
    else:
        shaft_dir = _normalize(left_hand - right_hand)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('output_dir', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    kps_path = os.path.join(args.output_dir, 'keypoints.json')
    video_path = os.path.join(args.output_dir, 'robot_golf.mp4')

    _, fps = extract_keypoints(args.video, kps_path)
    render_skeleton_video(kps_path, video_path, fps=fps)
    print(f"Skeleton pipeline done: {kps_path}, {video_path}")
