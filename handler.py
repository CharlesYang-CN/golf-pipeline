"""RunPod Serverless Handler for Golf Swing Analysis Pipeline."""
import json, os, subprocess, sys, time, urllib.request, uuid, traceback

print("[handler] Python:", sys.version, flush=True)
print("[handler] CWD:", os.getcwd(), flush=True)

try:
    import runpod
    print("[handler] runpod imported OK", flush=True)
except Exception as e:
    print(f"[handler] FATAL: runpod import failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

SCRIPTS_DIR = '/app/scripts'
DATA_DIR = '/data'


def download_video(url, local_path):
    print(f"[handler] Downloading: {url[:80]}...", flush=True)
    urllib.request.urlretrieve(url, local_path)
    print(f"[handler] Downloaded to {local_path} ({os.path.getsize(local_path)} bytes)", flush=True)


def handler(event):
    job_input = event.get('input', {})
    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing 'video_url'"}

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, 'video.mp4')
    download_video(video_url, video_path)

    results = {"job_id": job_id}
    steps = {
        "swing":    ["swing", video_path, os.path.join(job_dir, "swing", "events.json")],
        "skeleton": ["skeleton", video_path, os.path.join(job_dir, "skeleton")],
        "mesh":     ["mesh", video_path, os.path.join(job_dir, "mesh")],
    }

    for name, (script, *args) in steps.items():
        dirpath = os.path.join(job_dir, name)
        os.makedirs(dirpath, exist_ok=True)
        cmd = ["python", os.path.join(SCRIPTS_DIR, f"run_{script}.py")] + args
        print(f"[handler] {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"[handler] {name}: done in {time.time()-t0:.0f}s, rc={r.returncode}", flush=True)
        if r.returncode != 0:
            return {"error": f"{name} failed", "stderr": r.stderr[-500:]}

    for name in steps:
        dirpath = os.path.join(job_dir, name)
        for fname in os.listdir(dirpath):
            if fname.endswith('.json'):
                with open(os.path.join(dirpath, fname)) as f:
                    results[name] = json.load(f)

    return results


print("[handler] Starting runpod.serverless.start()...", flush=True)
runpod.serverless.start({"handler": handler})
