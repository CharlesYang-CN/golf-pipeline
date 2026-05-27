import json
import cv2
import numpy as np
import open3d as o3d
import sys
import math

sys.path.insert(0, "/workspace/UPose/MotionCapture/upose")
from upose import UPose

KPS_PATH = "/workspace/angle/keypoints.json"
OUTPUT_PATH = "/workspace/angle/robot_3d_golf.mp4"

W, H = 1280, 720

class Landmark:
    def __init__(self, x, y, z, visibility):
        self.x = x; self.y = y; self.z = z; self.visibility = visibility
class LandmarkList:
    def __init__(self, landmarks):
        self.landmark = [Landmark(*lm) for lm in landmarks]
class Results:
    def __init__(self, landmarks):
        self.pose_world_landmarks = LandmarkList(landmarks) if landmarks is not None else None

def load_kps(path):
    with open(path) as f:
        data = json.load(f)
    fps = data.get("fps", 30)
    frames = []
    for entry in data["data"]:
        pts = []
        for p in entry["keypoints"]:
            if isinstance(p, dict):
                pts.append([p["x"], p["y"], p["z"], p.get("visibility", 1.0)])
            else:
                pts.append([p[0], p[1], p[2] if len(p) > 2 else 0, p[3] if len(p) > 3 else 1.0])
        frames.append(pts)
    return np.array(frames, dtype=np.float32), fps

kps, fps = load_kps(KPS_PATH)
print(f"Loaded {len(kps)} frames, fps={fps}")

pose_tracker = UPose(source="mediapipe", flipped=False)

# Build 3D robot from primitives
def make_bone(length, radius=0.04, color=(0.3, 0.6, 0.9)):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    mesh.translate([0, length/2, 0])
    return mesh

def make_sphere(radius=0.06, color=(0.9, 0.9, 0.2)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def make_head(radius=0.1, color=(0.9, 0.5, 0.2)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

BONE_LENGTHS = {
    "pelvis": 0.2, "torso": 0.35,
    "shoulder": 0.2, "upper_arm": 0.25, "forearm": 0.25,
    "thigh": 0.3, "leg": 0.3,
}

renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
renderer.scene.set_background([0.05, 0.05, 0.08, 1.0])

# Camera setup
cam = renderer.scene.camera
cam.look_at([0, 0.8, 0], [2, 0.8, 2], [0, 1, 0])

# Lighting
renderer.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, np.array([1, 2, 3], dtype=np.float32))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

for t in range(len(kps)):
    pts = kps[t]
    results = Results(pts)
    pose_tracker.newFrame(results)
    pose_tracker.computeRotations()
    angles = pose_tracker.getAngleVector(format="euler")

    # pelvis 0: y-rotation
    # torso 1:x, 2:z
    # l_shoulder 3:x,4:y,5:z
    # r_shoulder 6:x,7:y,8:z
    # l_elbow 9:z
    # r_elbow 10:z
    # l_hip 11:x,12:z
    # r_hip 13:x,14:z
    # l_knee 15:x,16:z
    # r_knee 17:x,18:z

    # Build scene hierarchy in Open3D
    # Since Open3D doesn't have a scene graph, we manually transform each mesh
    scene_meshes = []

    # Pelvis
    pelvis_rot = angles[0]
    pelvis_pos = np.array([0, 0.8, 0])

    # Torso
    torso_rot_x = math.radians(angles[1])
    torso_rot_z = math.radians(angles[2])
    r_y = pelvis_rot

    # Helper: apply euler ZXY
    def euler_zxy(e):
        z, x, y = math.radians(e[2]), math.radians(e[0]), math.radians(e[1])
        Rz = np.array([[math.cos(z), -math.sin(z), 0],
                       [math.sin(z),  math.cos(z), 0],
                       [0, 0, 1]])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(x), -math.sin(x)],
                       [0, math.sin(x),  math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)],
                       [0, 1, 0],
                       [-math.sin(y), 0, math.cos(y)]])
        return Rz @ Rx @ Ry

    # Pelvis bone (hips segment)
    pelvis_mesh = make_bone(0.15, 0.06, (0.2, 0.5, 0.8))
    pelvis_mesh.rotate(euler_zxy([0, 0, r_y]), center=(0,0,0))
    pelvis_mesh.translate(pelvis_pos)
    scene_meshes.append(pelvis_mesh)
    scene_meshes.append(make_sphere(0.07, (0.8, 0.8, 0.2)).translate(pelvis_pos))

    # Torso
    torso_rot = euler_zxy([torso_rot_z, torso_rot_x, 0])
    torso_mesh = make_bone(0.35, 0.07, (0.3, 0.6, 0.9))
    torso_mesh.rotate(torso_rot, center=(0,0,0))
    torso_mesh.rotate(euler_zxy([0, 0, r_y]), center=(0,0,0))
    torso_mesh.translate(pelvis_pos)
    scene_meshes.append(torso_mesh)
    torso_end = pelvis_pos + torso_rot @ euler_zxy([0, 0, r_y]) @ np.array([0, 0.35, 0])

    # Shoulder center
    shoulder_center = torso_end

    # Left shoulder
    l_shoulder_rot = euler_zxy([angles[5], angles[3], angles[4]])
    l_shoulder_mesh = make_bone(0.18, 0.04, (0.6, 0.3, 0.8))
    l_shoulder_mesh.rotate(l_shoulder_rot, center=(0,0,0))
    l_shoulder_mesh.rotate(torso_rot @ euler_zxy([0, 0, r_y]), center=(0,0,0))
    l_shoulder_mesh.translate(shoulder_center)
    scene_meshes.append(l_shoulder_mesh)
    l_shoulder_end = shoulder_center + (torso_rot @ euler_zxy([0, 0, r_y])) @ l_shoulder_rot @ np.array([0, 0.18, 0])

    # Right shoulder
    r_shoulder_rot = euler_zxy([angles[8], angles[6], angles[7]])
    r_shoulder_mesh = make_bone(0.18, 0.04, (0.6, 0.3, 0.8))
    r_shoulder_mesh.rotate(r_shoulder_rot, center=(0,0,0))
    r_shoulder_mesh.rotate(torso_rot @ euler_zxy([0, 0, r_y]), center=(0,0,0))
    r_shoulder_mesh.translate(shoulder_center)
    scene_meshes.append(r_shoulder_mesh)
    r_shoulder_end = shoulder_center + (torso_rot @ euler_zxy([0, 0, r_y])) @ r_shoulder_rot @ np.array([0, 0.18, 0])

    # Left elbow
    l_elbow_angle = math.radians(abs(angles[9]))
    l_elbow_rot = euler_zxy([l_elbow_angle, 0, 0])
    l_forearm_mesh = make_bone(0.25, 0.035, (0.8, 0.4, 0.3))
    l_forearm_mesh.rotate(l_elbow_rot, center=(0,0,0))
    world_l_shoulder = (torso_rot @ euler_zxy([0, 0, r_y])) @ l_shoulder_rot
    l_forearm_mesh.rotate(world_l_shoulder, center=(0,0,0))
    l_forearm_mesh.translate(l_shoulder_end)
    scene_meshes.append(l_forearm_mesh)
    scene_meshes.append(make_sphere(0.045, (0.9, 0.7, 0.2)).translate(l_shoulder_end))

    # Right elbow
    r_elbow_angle = math.radians(abs(angles[10]))
    r_elbow_rot = euler_zxy([r_elbow_angle, 0, 0])
    r_forearm_mesh = make_bone(0.25, 0.035, (0.8, 0.4, 0.3))
    r_forearm_mesh.rotate(r_elbow_rot, center=(0,0,0))
    world_r_shoulder = (torso_rot @ euler_zxy([0, 0, r_y])) @ r_shoulder_rot
    r_forearm_mesh.rotate(world_r_shoulder, center=(0,0,0))
    r_forearm_mesh.translate(r_shoulder_end)
    scene_meshes.append(r_forearm_mesh)
    scene_meshes.append(make_sphere(0.045, (0.9, 0.7, 0.2)).translate(r_shoulder_end))

    # Left thigh
    l_hip_rot = euler_zxy([angles[12], angles[11], 0])
    l_thigh_mesh = make_bone(0.3, 0.05, (0.2, 0.5, 0.7))
    l_thigh_mesh.rotate(l_hip_rot, center=(0,0,0))
    l_thigh_mesh.rotate(euler_zxy([0, 0, r_y]), center=(0,0,0))
    l_thigh_mesh.translate(pelvis_pos - np.array([0.08, 0, 0.05]))
    scene_meshes.append(l_thigh_mesh)
    l_knee_pos = pelvis_pos - np.array([0.08, 0, 0.05]) + euler_zxy([0, 0, r_y]) @ l_hip_rot @ np.array([0, 0.3, 0])

    # Right thigh
    r_hip_rot = euler_zxy([angles[14], angles[13], 0])
    r_thigh_mesh = make_bone(0.3, 0.05, (0.2, 0.5, 0.7))
    r_thigh_mesh.rotate(r_hip_rot, center=(0,0,0))
    r_thigh_mesh.rotate(euler_zxy([0, 0, r_y]), center=(0,0,0))
    r_thigh_mesh.translate(pelvis_pos + np.array([0.08, 0, -0.05]))
    scene_meshes.append(r_thigh_mesh)
    r_knee_pos = pelvis_pos + np.array([0.08, 0, -0.05]) + euler_zxy([0, 0, r_y]) @ r_hip_rot @ np.array([0, 0.3, 0])

    # Left knee
    l_knee_rot = euler_zxy([angles[16], angles[15], 0])
    l_leg_mesh = make_bone(0.3, 0.04, (0.7, 0.5, 0.2))
    l_leg_mesh.rotate(l_knee_rot, center=(0,0,0))
    l_leg_mesh.rotate(euler_zxy([0, 0, r_y]) @ l_hip_rot, center=(0,0,0))
    l_leg_mesh.translate(l_knee_pos)
    scene_meshes.append(l_leg_mesh)
    scene_meshes.append(make_sphere(0.04, (0.8, 0.8, 0.2)).translate(l_knee_pos))

    # Right knee
    r_knee_rot = euler_zxy([angles[18], angles[17], 0])
    r_leg_mesh = make_bone(0.3, 0.04, (0.7, 0.5, 0.2))
    r_leg_mesh.rotate(r_knee_rot, center=(0,0,0))
    r_leg_mesh.rotate(euler_zxy([0, 0, r_y]) @ r_hip_rot, center=(0,0,0))
    r_leg_mesh.translate(r_knee_pos)
    scene_meshes.append(r_leg_mesh)
    scene_meshes.append(make_sphere(0.04, (0.8, 0.8, 0.2)).translate(r_knee_pos))

    # Head
    head_mesh = make_head(0.12, (0.9, 0.3, 0.2))
    head_mesh.translate(shoulder_center + np.array([0, 0.15, 0]))
    scene_meshes.append(head_mesh)

    # Clear and render
    renderer.scene.clear_geometry()
    for i, mesh in enumerate(scene_meshes):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry(f"mesh_{i}", mesh, mat)

    img = renderer.render_to_image()
    frame = np.asarray(img)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Overlay info
    info = [
        f"Frame: {t}",
        f"Pelvis: {angles[0]:6.1f}",
        f"L-Elbow: {angles[9]:5.1f}  R-Elbow: {angles[10]:5.1f}",
        f"L-Knee:  {angles[15]:5.1f}  R-Knee:  {angles[17]:5.1f}",
    ]
    for i, line in enumerate(info):
        cv2.putText(frame_bgr, line, (16, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, "3D ROBOT GOLF - UPose", (16, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2, cv2.LINE_AA)

    writer.write(frame_bgr)
    if (t + 1) % 30 == 0 or t == len(kps) - 1:
        print(f"Rendered {t + 1}/{len(kps)}")

writer.release()
print(f"Saved to: {OUTPUT_PATH}")
