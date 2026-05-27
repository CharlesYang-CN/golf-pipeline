import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image as MPImage, ImageFormat
import numpy as np
import sys
import os

sys.path.insert(0, "/workspace/UPose/MotionCapture/upose")
from upose import UPose

VIDEO_PATH = "/workspace/angle/24.mp4"
MODEL_PATH = "/workspace/angle/pose_landmarker.task"
OUTPUT_PATH = "/workspace/angle/24_upose.mp4"

class LandmarkWrapper:
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z
        self.visibility = getattr(lm, 'visibility', 1.0)

class WorldLandmarksWrapper:
    def __init__(self, landmarks):
        self.landmark = [LandmarkWrapper(lm) for lm in landmarks]

class ResultsWrapper:
    def __init__(self, landmarks):
        if landmarks:
            self.pose_world_landmarks = WorldLandmarksWrapper(landmarks)
        else:
            self.pose_world_landmarks = None

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
    (11, 23), (12, 24),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)

pose_tracker = UPose(source="mediapipe", flipped=False)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, frame_idx)
    canvas = frame.copy()

    landmarks_list = None
    if result.pose_landmarks:
        landmarks_list = result.pose_landmarks[0]
        h, w, _ = frame.shape
        for connection in POSE_CONNECTIONS:
            i1, i2 = connection
            if i1 < len(landmarks_list) and i2 < len(landmarks_list):
                p1 = (int(landmarks_list[i1].x * w), int(landmarks_list[i1].y * h))
                p2 = (int(landmarks_list[i2].x * w), int(landmarks_list[i2].y * h))
                cv2.line(canvas, p1, p2, (0, 255, 100), 2, cv2.LINE_AA)
        for i, lm in enumerate(landmarks_list):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    wrapped = ResultsWrapper(landmarks_list)
    if wrapped.pose_world_landmarks:
        pose_tracker.newFrame(wrapped)
        pose_tracker.computeRotations()
        angles = pose_tracker.getAngleVector(format="euler")

        info_lines = [
            f"Frame: {frame_idx}",
            f"Pelvis: {angles[0]:.1f}",
            f"Torso: ({angles[1]:.1f}, {angles[2]:.1f})",
            f"L-Shoulder: ({angles[3]:.1f}, {angles[4]:.1f}, {angles[5]:.1f})",
            f"R-Shoulder: ({angles[6]:.1f}, {angles[7]:.1f}, {angles[8]:.1f})",
            f"L-Elbow: {angles[9]:.1f}",
            f"R-Elbow: {angles[10]:.1f}",
            f"L-Hip: ({angles[11]:.1f}, {angles[12]:.1f})",
            f"R-Hip: ({angles[13]:.1f}, {angles[14]:.1f})",
            f"L-Knee: ({angles[15]:.1f}, {angles[16]:.1f})",
            f"R-Knee: ({angles[17]:.1f}, {angles[18]:.1f})",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(canvas, line, (16, 30 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)

    cv2.putText(canvas, "UPose Motion Tracking", (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2, cv2.LINE_AA)

    writer.write(canvas)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames")

cap.release()
landmarker.close()
writer.release()
print(f"Done. Saved to: {OUTPUT_PATH}")
print(f"Total frames: {frame_idx}")
