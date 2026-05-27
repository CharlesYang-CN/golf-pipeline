import json
import sys
import os
from collections import deque

import cv2
import numpy as np


NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LEFT_HAND_IDS = [LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB]
RIGHT_HAND_IDS = [RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB]

BONES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
]

HAND_BONES = [
    (LEFT_WRIST, LEFT_PINKY),
    (LEFT_WRIST, LEFT_INDEX),
    (LEFT_WRIST, LEFT_THUMB),
    (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_WRIST, RIGHT_INDEX),
    (RIGHT_WRIST, RIGHT_THUMB),
]


def valid_point(pts, idx, vis_thr=0.15):
    return pts[idx, 2] >= vis_thr


def get_point(pts, idx):
    return pts[idx, :2].astype(np.float32)


def get_point_int(pts, idx):
    x, y = pts[idx, :2]
    return int(round(x)), int(round(y))


def mean_visible_points(pts, ids, vis_thr=0.15):
    arr = [get_point(pts, i) for i in ids if valid_point(pts, i, vis_thr)]
    if not arr:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.mean(np.stack(arr, axis=0), axis=0)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def estimate_body_height(pts, default_h=500):
    vis = pts[:, 2] > 0.15
    if np.sum(vis) < 4:
        return default_h
    ys = pts[vis, 1]
    return float(np.max(ys) - np.min(ys))


def estimate_club(pts, handedness="right", default_h=500):
    left_hand = mean_visible_points(pts, LEFT_HAND_IDS)
    right_hand = mean_visible_points(pts, RIGHT_HAND_IDS)
    grip = (left_hand + right_hand) / 2.0

    if handedness == "right":
        shaft_dir = normalize(right_hand - left_hand)
    else:
        shaft_dir = normalize(left_hand - right_hand)

    if shaft_dir is None:
        shaft_dir = np.array([1.0, 0.0], dtype=np.float32)

    body_h = estimate_body_height(pts, default_h=default_h)
    club_length = 0.60 * body_h

    butt_end = grip - 0.12 * club_length * shaft_dir
    club_head = grip + 0.88 * club_length * shaft_dir

    return grip, butt_end, club_head


def draw_glow_line(img, p1, p2, color, thickness=4, glow_thickness=16, glow_alpha=0.18):
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, glow_thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, glow_alpha, img, 1 - glow_alpha, 0, img)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_glow_circle(img, center, radius, color, glow_radius=None, glow_alpha=0.18, thickness=-1):
    if glow_radius is None:
        glow_radius = radius + 8
    overlay = img.copy()
    cv2.circle(overlay, center, glow_radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, glow_alpha, img, 1 - glow_alpha, 0, img)
    cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)


def draw_torso(img, pts):
    if not (valid_point(pts, LEFT_SHOULDER) and valid_point(pts, RIGHT_SHOULDER) and
            valid_point(pts, LEFT_HIP) and valid_point(pts, RIGHT_HIP)):
        return

    ls = get_point_int(pts, LEFT_SHOULDER)
    rs = get_point_int(pts, RIGHT_SHOULDER)
    lh = get_point_int(pts, LEFT_HIP)
    rh = get_point_int(pts, RIGHT_HIP)

    poly = np.array([ls, rs, rh, lh], dtype=np.int32)

    overlay = img.copy()
    cv2.fillConvexPoly(overlay, poly, (70, 80, 100), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.polylines(img, [poly], isClosed=True, color=(180, 220, 255), thickness=2, lineType=cv2.LINE_AA)

    center = np.mean(poly, axis=0).astype(np.int32)
    draw_glow_circle(img, tuple(center), 8, (255, 200, 50), glow_radius=20, glow_alpha=0.24)


def draw_head(img, pts):
    base = []
    for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
        if valid_point(pts, idx):
            base.append(get_point(pts, idx))
    if not base:
        return

    base = np.stack(base, axis=0)
    center = np.mean(base, axis=0)

    radius = 18
    if valid_point(pts, LEFT_EAR) and valid_point(pts, RIGHT_EAR):
        ear_dist = np.linalg.norm(get_point(pts, LEFT_EAR) - get_point(pts, RIGHT_EAR))
        radius = max(16, int(0.35 * ear_dist))
    elif valid_point(pts, LEFT_EYE) and valid_point(pts, RIGHT_EYE):
        eye_dist = np.linalg.norm(get_point(pts, LEFT_EYE) - get_point(pts, RIGHT_EYE))
        radius = max(14, int(0.8 * eye_dist))

    c = tuple(center.astype(np.int32))

    overlay = img.copy()
    cv2.circle(overlay, c, radius, (85, 95, 110), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    draw_glow_circle(img, c, radius, (160, 230, 255), glow_radius=radius + 10, glow_alpha=0.14, thickness=2)

    if valid_point(pts, LEFT_EYE) and valid_point(pts, RIGHT_EYE):
        le = get_point_int(pts, LEFT_EYE)
        re = get_point_int(pts, RIGHT_EYE)
        draw_glow_line(img, le, re, (0, 255, 255), thickness=3, glow_thickness=12, glow_alpha=0.22)


def draw_joint(img, p, joint_radius=5):
    x, y = int(round(p[0])), int(round(p[1]))
    draw_glow_circle(img, (x, y), joint_radius, (220, 240, 255), glow_radius=joint_radius + 8, glow_alpha=0.20)
    cv2.circle(img, (x, y), joint_radius - 2, (120, 160, 220), -1, cv2.LINE_AA)


def draw_limb_robot(img, p1, p2, base_color=(180, 220, 255), core_color=(90, 130, 200)):
    p1 = tuple(np.int32(np.round(p1)))
    p2 = tuple(np.int32(np.round(p2)))

    draw_glow_line(img, p1, p2, base_color, thickness=10, glow_thickness=22, glow_alpha=0.10)
    cv2.line(img, p1, p2, (80, 95, 120), 8, cv2.LINE_AA)
    cv2.line(img, p1, p2, core_color, 3, cv2.LINE_AA)


def draw_swing_trail(img, trail_points):
    if len(trail_points) < 2:
        return
    pts = list(trail_points)
    n = len(pts)

    for i in range(1, n):
        p1 = tuple(np.int32(np.round(pts[i - 1])))
        p2 = tuple(np.int32(np.round(pts[i])))
        w = i / (n - 1 + 1e-6)
        thickness = max(1, int(2 + 6 * w))
        color = (int(40 + 215 * w), int(100 + 155 * w), 255)

        overlay = img.copy()
        cv2.line(overlay, p1, p2, color, thickness + 8, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.08 + 0.12 * w, img, 1 - (0.08 + 0.12 * w), 0, img)
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def to_pixel_coords(frame_kps, width, height):
    pts = frame_kps.copy()
    max_xy = np.nanmax(pts[:, :2])
    if max_xy <= 1.5:
        pts[:, 0] *= width
        pts[:, 1] *= height
    return pts


def load_kps_json(path):
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
                pts.append([p.get("x", 0.0), p.get("y", 0.0), p.get("visibility", 1.0)])
            else:
                pts.append(p[:3] if len(p) >= 3 else [p[0], p[1], 1.0])
        frames.append(pts)

    return np.array(frames, dtype=np.float32), fps


def render_frame(width, height, pts, handedness="right", trail_points=None):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        v = int(8 + 18 * (y / height))
        canvas[y, :, :] = (v, v, v + 8)

    if trail_points is not None:
        draw_swing_trail(canvas, trail_points)

    draw_torso(canvas, pts)
    draw_head(canvas, pts)

    for a, b in BONES:
        if valid_point(pts, a) and valid_point(pts, b):
            draw_limb_robot(canvas, get_point(pts, a), get_point(pts, b))

    for a, b in HAND_BONES:
        if valid_point(pts, a) and valid_point(pts, b):
            draw_limb_robot(canvas, get_point(pts, a), get_point(pts, b),
                            base_color=(160, 255, 220), core_color=(80, 180, 130))

    joint_ids = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    ]
    for j in joint_ids:
        if valid_point(pts, j):
            draw_joint(canvas, get_point(pts, j), joint_radius=6)

    grip, butt_end, club_head = estimate_club(pts, handedness=handedness, default_h=height * 0.65)
    p_grip = tuple(np.int32(np.round(grip)))
    p_butt = tuple(np.int32(np.round(butt_end)))
    p_head = tuple(np.int32(np.round(club_head)))

    draw_glow_line(canvas, p_butt, p_head, (50, 220, 255), thickness=3, glow_thickness=12, glow_alpha=0.22)
    draw_glow_circle(canvas, p_grip, 6, (255, 200, 50), glow_radius=14, glow_alpha=0.28)
    draw_glow_circle(canvas, p_head, 7, (255, 120, 60), glow_radius=18, glow_alpha=0.30)

    cv2.putText(canvas, "ROBOT GOLF SWING", (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (180, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "POSE-DRIVEN VISUALIZATION", (34, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (120, 180, 220), 1, cv2.LINE_AA)

    return canvas, club_head


def main():
    kps_path = "/workspace/angle/keypoints.json"
    out_path = "/workspace/angle/robot_golf.mp4"

    kps, fps = load_kps_json(kps_path)
    print(f"Loaded {len(kps)} frames, fps={fps}")

    width, height = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    trail = deque(maxlen=18)

    for t in range(len(kps)):
        pts = to_pixel_coords(kps[t], width, height)

        frame_img, club_head = render_frame(width, height, pts, trail_points=trail)
        trail.append(club_head.copy())
        writer.write(frame_img)

        if (t + 1) % 30 == 0 or t == len(kps) - 1:
            print(f"Rendered {t + 1}/{len(kps)}")

    writer.release()
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
