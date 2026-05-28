"""Modal deployment for Golf Swing Analysis Pipeline.

Deploy:
  pip install modal
  modal token new          # or: modal login
  modal deploy modal_app.py

Test:
  modal run modal_app.py   # run locally-on-cloud test

API Call:
  curl -X POST https://YOUR_USER--golf-pipeline-analyze.modal.run \
    -H "Content-Type: application/json" \
    -d '{"video_url": "https://example.com/golf.mp4"}'
"""
import json
import os
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path

import modal

GPU_CONFIG = modal.gpu.T4(count=1)

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6",
        "libxrender-dev", "libgomp1", "libegl1-mesa", "libgl1",
        "ffmpeg", "git", "build-essential", "wget",
    )
    # Copy source code BEFORE pip install -e (needed for ROMP Cython build)
    .add_local_dir(HERE / "romp", remote_path="/app/romp")
    .add_local_dir(HERE / "angle", remote_path="/app/angle")
    .add_local_dir(HERE / "swing", remote_path="/app/swing")
    .add_local_dir(HERE / "romp_models", remote_path="/root/.romp")
    .add_local_file(HERE / "scripts" / "run_swing.py", remote_path="/app/scripts/run_swing.py")
    .add_local_file(HERE / "scripts" / "run_skeleton.py", remote_path="/app/scripts/run_skeleton.py")
    .add_local_file(HERE / "scripts" / "run_mesh.py", remote_path="/app/scripts/run_mesh.py")
    # PyTorch (CUDA)
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # All Python deps + ROMP Cython extension
    .pip_install(
        "mediapipe", "opencv-python", "numpy==1.26.4", "scipy>=1.11",
        "cython", "lapx", "wget", "ultralytics",
    )
    .run_commands("pip install --no-build-isolation -e /app/romp/")
)

app = modal.App("golf-pipeline", image=image)


@app.cls(gpu=GPU_CONFIG, container_idle_timeout=300)
class GolfPipeline:
    @modal.enter()
    def startup(self):
        import torch
        import sys
        sys.path.insert(0, "/app/swing")
        sys.path.insert(0, "/app/romp")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[startup] Device={self.device}, GPU={torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'}")

    @modal.method()
    def analyze(self, video_url: str):
        job_id = str(uuid.uuid4())[:8]
        out_dir = f"/tmp/{job_id}"
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, "video.mp4")
        urllib.request.urlretrieve(video_url, video_path)

        results = {}

        # 1. Swing Phase Detection
        d = os.path.join(out_dir, "swing"); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "events.json")
        r = subprocess.run(["python", "/app/scripts/run_swing.py", video_path, p],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return {"error": f"swing: {r.stderr[-300:]}"}
        with open(p) as f:
            results["swing_events"] = json.load(f)

        # 2. Skeleton Extraction
        d = os.path.join(out_dir, "skeleton"); os.makedirs(d, exist_ok=True)
        r = subprocess.run(["python", "/app/scripts/run_skeleton.py", video_path, d],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return {"error": f"skeleton: {r.stderr[-300:]}"}
        with open(os.path.join(d, "keypoints.json")) as f:
            results["skeleton_keypoints"] = json.load(f)

        # 3. 3D Mesh Recovery
        d = os.path.join(out_dir, "mesh"); os.makedirs(d, exist_ok=True)
        r = subprocess.run(["python", "/app/scripts/run_mesh.py", video_path, d],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return {"error": f"mesh: {r.stderr[-300:]}"}
        with open(os.path.join(d, "keypoints3d.json")) as f:
            results["mesh_3d_joints"] = json.load(f)

        results["job_id"] = job_id
        return results


@app.local_entrypoint()
def main():
    """Test: modal run modal_app.py"""
    pipeline = GolfPipeline()
    result = pipeline.analyze.remote(
        "https://github.com/CharlesYang-CN/golf-pipeline/releases/download/test-assets/24.mp4"
    )
    if "error" in result:
        print("ERROR:", result["error"])
    else:
        for name, data in result.items():
            if name == "job_id":
                continue
            print(f"\n{name}:")
            if "events" in data:
                for e in data["events"]:
                    print(f"  {e['event']}: frame={e['frame']}, time={e['time_s']}s, conf={e['confidence']}")
            elif "data" in data:
                print(f"  frames: {data.get('total_frames', len(data['data']))}")
