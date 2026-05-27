"""RunPod Serverless Handler for Golf Swing Analysis Pipeline.

Input: {"input": {"video_url": "https://example.com/golf_swing.mp4"}}
Output: swing events + skeleton keypoints + 3D mesh joints + video paths
"""
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import uuid

import runpod

SCRIPTS_DIR = '/app/scripts'
DATA_DIR = '/data'


def download_video(url, local_path):
    print(f"[handler] Downloading video from {url}...")
    urllib.request.urlretrieve(url, local_path)
    print(f"[handler] Downloaded to {local_path}")
    return local_path


def run_step(name, args, timeout=900):
    cmd = ['python', os.path.join(SCRIPTS_DIR, f'run_{name}.py')] + args
    print(f"[handler] Running {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print(f"[handler] {name} stdout:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"[handler] {name} stderr:", result.stderr[-500:])
        raise RuntimeError(f"{name} failed: {result.stderr[-200:]}")
    return result


def handler(event):
    job_input = event.get('input', {})
    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing 'video_url' in input"}

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, 'input_video.mp4')
    download_video(video_url, video_path)

    swings = {'swing': os.path.join(job_dir, 'swing'),
              'skeleton': os.path.join(job_dir, 'skeleton'),
              'mesh': os.path.join(job_dir, 'mesh')}
    for d in swings.values():
        os.makedirs(d, exist_ok=True)

    start = time.time()
    run_step('swing', [video_path, os.path.join(swings['swing'], 'events.json')])
    print(f"[handler] Swing: {time.time() - start:.0f}s")

    start = time.time()
    run_step('skeleton', [video_path, swings['skeleton']])
    print(f"[handler] Skeleton: {time.time() - start:.0f}s")

    start = time.time()
    run_step('mesh', [video_path, swings['mesh']])
    print(f"[handler] Mesh: {time.time() - start:.0f}s")

    result = {'job_id': job_id}

    for name, dirpath in swings.items():
        for fname in os.listdir(dirpath):
            if fname.endswith('.json'):
                with open(os.path.join(dirpath, fname)) as f:
                    result[name] = json.load(f)

    videos = {}
    for name, dirpath in swings.items():
        for fname in os.listdir(dirpath):
            if fname.endswith('.mp4'):
                videos[name] = os.path.join(dirpath, fname)
    if videos:
        result['videos'] = videos

    return result


if __name__ == '__main__':
    print("[handler] Starting Golf Pipeline handler...")
    runpod.serverless.start({"handler": handler})
