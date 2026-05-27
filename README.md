# Golf Swing Analysis Pipeline

End-to-end golf swing video analysis deployed on RunPod Serverless.

**Input** → a golf swing video URL  
**Output** → skeleton keypoints, swing event timestamps, and 3D mesh video

---

## Pipeline Architecture

```
  Video URL
      │
  ┌───▼───────┐   ┌──────────────┐   ┌────────────┐
  │  Swing    │   │   Skeleton   │   │  3D Mesh   │
  │  Phase    │   │   (angle)    │   │   (ROMP)   │
  │  Detection│   └──────────────┘   └────────────┘
  └───┬───────┘         │                  │
      │                 │                  │
  8 events         33 keypoints        71 joints
  + timestamps     + 2D skeleton       + 3D mesh
  + confidence     animation video     overlay video
      │                 │                  │
      └─────────────────┴──────────────────┘
                        │
                   JSON response
```

## Three Models

| Pipeline | Model | Output |
|----------|-------|--------|
| **Swing Phase Detection** | SwingNet (MobileNetV2 + BiLSTM) | 8 golf swing events: Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish |
| **Skeleton Extraction** | MediaPipe Pose Landmarker | 33 body keypoints per frame (x, y, z, visibility) + 2D visualization video |
| **3D Mesh Recovery** | ROMP (HRNet + SMPL) | 71 joints in 3D + 3D mesh overlay video |

## Quick Start (RunPod Serverless)

### 1. Image

```
ghcr.io/charlesyang-cn/golf-pipeline:latest
```

### 2. Create Endpoint

| Setting | Value |
|---------|-------|
| Container Image | `ghcr.io/charlesyang-cn/golf-pipeline:latest` |
| GPU | NVIDIA RTX 3090 (24GB) |
| Min Workers | 0 |
| Max Workers | 3 |
| Idle Timeout | 300 |
| Container Disk | 50 GB |

### 3. Make a Request

**Input:**
```json
{
  "input": {
    "video_url": "https://example.com/golf-swing.mp4"
  }
}
```

**Output:**
```json
{
  "swing_events": {
    "fps": 30.0,
    "total_frames": 218,
    "events": [
      {"event": "Address", "frame": 23, "time_s": 0.77, "confidence": 0.32},
      {"event": "Impact", "frame": 128, "time_s": 4.27, "confidence": 0.45}
    ]
  },
  "skeleton_keypoints": {
    "data": [{"frame": 0, "keypoints": [{"x": 0.46, "y": 0.45, "z": -0.76, "visibility": 0.99}]}]
  },
  "mesh_3d_joints": {
    "data": [{"frame": 0, "persons": [{"joints_3d": [[...]], "joints_2d": [[...]]}]}]
  },
  "videos": {
    "skeleton_2d": "/data/<job_id>/skeleton/robot_golf.mp4",
    "mesh_3d": "/data/<job_id>/mesh/mesh.mp4"
  }
}
```

### 4. Call from Code

```python
import requests, json

resp = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={"input": {"video_url": "https://example.com/golf.mp4"}},
    timeout=300
)
print(json.dumps(resp.json(), indent=2))
```

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "https://example.com/golf.mp4"}}'
```

## Local Development

### Prerequisites

- NVIDIA GPU with >= 8GB VRAM
- Docker or Python 3.10 + pip

### Build Docker Image

```bash
bash build.sh
docker build -t golf-pipeline:latest .
```

### Test Locally

```bash
# With conda environments (optional)
conda env create -f env_swing.yml
conda env create -f env_angle.yml
conda env create -f env_romp.yml

# Or with pip (single environment)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install mediapipe opencv-python numpy==1.26.4 scipy cython lapx wget ultralytics runpod
pip install -e romp/

# Run pipelines
python scripts/run_swing.py input.mp4 events.json
python scripts/run_skeleton.py input.mp4 output_dir/
python scripts/run_mesh.py input.mp4 output_dir/

# Run handler
python handler.py
```

## GitHub Actions CI/CD

Push to `main` triggers automatic Docker build and push to `ghcr.io/charlesyang-cn/golf-pipeline:latest`.

## Dependencies

```text
Python 3.10, PyTorch 2.5.1+cu124, MediaPipe, OpenCV, NumPy 1.26
Ultralytics, Cython, SciPy, RunPod SDK
ROMP (simple_romp), SwingNet (MobileNetV2 + BiLSTM)
```

## Credits

- **ROMP** - Yu Sun et al. (ICCV 2021) - [github.com/Arthur151/ROMP](https://github.com/Arthur151/ROMP)
- **SwingNet** - McNally et al. (CVPR 2019 Workshop) - Golf swing event detection
- **MediaPipe** - Google - Pose landmark detection
