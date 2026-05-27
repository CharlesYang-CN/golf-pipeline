import json
import cv2
import numpy as np
import sys

sys.path.insert(0, "/workspace/UPose/MotionCapture/upose")
from upose import UPose

KPS_PATH = "/workspace/angle/keypoints.json"
OUTPUT_PATH = "/workspace/angle/robot_golf_upose.mp4"

BONES = [
    (11, 12), (23, 24),
    (11, 23), (12, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

HAND_BONES = [
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

def load_kps(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("fps", 30)
    frames_raw = data["data"]
    frames = []
    for entry in frames_raw:
        kps = entry["keypoints"]
        pts = []
        for p in kps:
            if isinstance(p, dict):
                pts.append([p.get("x", 0), p.get("y", 0), p.get("z", 0), p.get("visibility", 1.0)])
            else:
                pts.append([p[0], p[1], p[2] if len(p) > 2 else 0, p[3] if len(p) > 3 else 1.0])
        frames.append(pts)
    return np.array(frames, dtype=np.float32), fps

class Landmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class LandmarkList:
    def __init__(self, landmarks):
        self.landmark = [Landmark(*lm) for lm in landmarks]

class Results:
    def __init__(self, landmarks):
        self.pose_world_landmarks = LandmarkList(landmarks) if landmarks is not None else None

kps, fps = load_kps(KPS_PATH)
print(f"Loaded {len(kps)} frames, fps={fps}")

pose_tracker = UPose(source="mediapipe", flipped=False)

W, H = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

def valid(pt, thr=0.15):
    return pt[3] >= thr

def draw_glow_line(img, p1, p2, color, thick=4, glow=16, alpha=0.18):
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, glow, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)

def draw_glow_circle(img, c, r, color, gr=None, alpha=0.18, thick=-1):
    if gr is None:
        gr = r + 8
    overlay = img.copy()
    cv2.circle(overlay, c, gr, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, c, r, color, thick, cv2.LINE_AA)

def draw_torso(canvas, pts):
    ids = [11, 12, 23, 24]
    if not all(valid(pts[i]) for i in ids):
        return
    poly = np.array([(int(pts[i][0]*W), int(pts[i][1]*H)) for i in ids], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, poly, (70, 80, 100), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.polylines(canvas, [poly], True, (180, 220, 255), 2, cv2.LINE_AA)

def draw_limb(canvas, p1, p2, base_color=(180, 220, 255), core_color=(90, 130, 200)):
    p1 = (int(p1[0]*W), int(p1[1]*H))
    p2 = (int(p2[0]*W), int(p2[1]*H))
    draw_glow_line(canvas, p1, p2, base_color, thick=10, glow=22, alpha=0.10)
    cv2.line(canvas, p1, p2, (80, 95, 120), 8, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, core_color, 3, cv2.LINE_AA)

def draw_hand_limb(canvas, p1, p2):
    p1 = (int(p1[0]*W), int(p1[1]*H))
    p2 = (int(p2[0]*W), int(p2[1]*H))
    draw_glow_line(canvas, p1, p2, (160, 255, 220), thick=6, glow=14, alpha=0.12)
    cv2.line(canvas, p1, p2, (80, 180, 130), 2, cv2.LINE_AA)

for t in range(len(kps)):
    pts = kps[t]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        v = int(8 + 18 * (y / H))
        canvas[y, :, :] = (v, v, v + 8)

    results = Results(pts)
    pose_tracker.newFrame(results)
    pose_tracker.computeRotations()
    angles = pose_tracker.getAngleVector(format="euler")

    draw_torso(canvas, pts)

    for a, b in BONES:
        if valid(pts[a]) and valid(pts[b]):
            draw_limb(canvas, pts[a], pts[b])

    for a, b in HAND_BONES:
        if valid(pts[a]) and valid(pts[b]):
            draw_hand_limb(canvas, pts[a], pts[b])

    joint_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for j in joint_ids:
        if valid(pts[j]):
            cx, cy = int(pts[j][0]*W), int(pts[j][1]*H)
            draw_glow_circle(canvas, (cx, cy), 6, (220, 240, 255), gr=14, alpha=0.20)
            cv2.circle(canvas, (cx, cy), 4, (120, 160, 220), -1, cv2.LINE_AA)

    cv2.putText(canvas, "UPose ROBOT GOLF SWING", (32, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 240, 255), 2, cv2.LINE_AA)

    info = [
        f"Pelvis: {angles[0]:6.1f}",
        f"Torso: ({angles[1]:5.1f}, {angles[2]:5.1f})",
        f"L-Elbow: {angles[9]:5.1f}   R-Elbow: {angles[10]:5.1f}",
        f"L-Knee: ({angles[15]:5.1f}, {angles[16]:5.1f})  R-Knee: ({angles[17]:5.1f}, {angles[18]:5.1f})",
    ]
    for i, line in enumerate(info):
        cv2.putText(canvas, line, (32, 80 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 180, 220), 1, cv2.LINE_AA)

    writer.write(canvas)
    if (t + 1) % 30 == 0 or t == len(kps) - 1:
        print(f"Rendered {t + 1}/{len(kps)}")

writer.release()
print(f"Saved to: {OUTPUT_PATH}")
