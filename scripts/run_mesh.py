"""Wrapper for ROMP mesh recovery on a golf swing video.

Usage: conda run -n env_romp python scripts/run_mesh.py <video_path> <output_dir>
Outputs:
  - output_dir/keypoints3d.json  (3D joints per frame)
  - output_dir/mesh.mp4          (3D mesh overlay video)
"""
import argparse
import json
import os
import sys
import cv2

sys.path.insert(0, '/app/romp')
import romp

def run_mesh(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    settings = romp.main.default_settings
    settings.mode = 'video'
    settings.calc_smpl = True
    settings.render_mesh = True
    settings.show_largest = True
    settings.GPU = 0

    model = romp.ROMP(settings)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames_data = []
    rendered_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = model(frame)
        frame_entry = {'frame': frame_idx, 'persons': []}
        if outputs is not None:
            joints = outputs.get('joints')
            if joints is not None:
                for p in range(joints.shape[0]):
                    person = {'joints_3d': joints[p].tolist()}
                    verts = outputs.get('verts')
                    pj2d = outputs.get('pj2d_org')
                    if verts is not None:
                        person['verts_count'] = verts[p].shape[0]
                    if pj2d is not None:
                        person['joints_2d'] = pj2d[p].tolist()
                    frame_entry['persons'].append(person)
            rendered = outputs.get('rendered_image')
            rendered_frames.append(rendered if rendered is not None else frame)
        else:
            rendered_frames.append(frame)
        all_frames_data.append(frame_entry)
        frame_idx += 1

    cap.release()

    with open(os.path.join(output_dir, 'keypoints3d.json'), 'w') as f:
        json.dump({
            'fps': float(fps),
            'total_frames': frame_idx,
            'width': w,
            'height': h,
            'data': all_frames_data
        }, f)

    if rendered_frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(output_dir, 'mesh.mp4')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for rf in rendered_frames:
            out.write(rf)
        out.release()
        print(f"Mesh video saved: {out_path}")

    print(f"3D keypoints saved: {os.path.join(output_dir, 'keypoints3d.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    run_mesh(args.video, args.output_dir)
