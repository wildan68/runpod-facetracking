import runpod
import os
import json
import urllib.request
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/yolov11n-face.pt")
MODEL_URL = (
    "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
)
MAX_DURATION_SEC = int(os.environ.get("MAX_DURATION_SEC", 7200))  # 2 hours
DEFAULT_FPS = 2.0
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.5

# ---------------------------------------------------------------------------
# Download model at cold start
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[init] Downloading model from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[init] Model saved to {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print(f"[init] Model loaded: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_video(url: str, dst: str):
    print(f"[download] {url} -> {dst}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; KlipYt-FaceTracking/1.0)",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as src, open(dst, "wb") as f:
        while True:
            chunk = src.read(8 * 1024 * 1024)  # 8MB
            if not chunk:
                break
            f.write(chunk)
    print(f"[download] Done")


def get_video_info(video_path: str) -> tuple:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_entries", "stream=width,height,codec_name,duration,r_frame_rate",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    orig_fps = float(num) / float(den)
    width = int(stream["width"])
    height = int(stream["height"])
    duration = float(stream.get("duration", 0))
    return orig_fps, width, height, duration


def process_video(video_path: str, target_fps: float, conf: float, iou: float) -> dict:
    orig_fps, width, height, duration = get_video_info(video_path)

    if duration > MAX_DURATION_SEC:
        raise RuntimeError(
            f"Video too long: {duration:.0f}s > {MAX_DURATION_SEC}s limit"
        )

    interval = max(1, round(orig_fps / target_fps))

    print(
        f"[process] {width}x{height}, {duration:.1f}s, "
        f"{int(duration * orig_fps)} frames @ {orig_fps:.2f}fps, interval={interval}"
    )

    # ── Extract frames via ffmpeg pipe (raw RGB24) ───────────────────────
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select=not(mod(n\\,{interval}))",
        "-vsync", "vfr",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )

    frame_size = width * height * 3
    face_timeline = []
    unique_track_ids: set[int] = set()
    frame_idx = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if not raw or len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        timestamp = frame_idx * interval / orig_fps

        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        detections = []
        boxes_data = results[0].boxes
        if boxes_data is not None and boxes_data.id is not None:
            boxes = boxes_data.xyxy.cpu().numpy()
            track_ids = boxes_data.id.cpu().numpy().astype(int)
            confs = boxes_data.conf.cpu().numpy()

            for box, tid, c in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "track_id": int(tid),
                        "bbox": [
                            round(x1 / width, 4),
                            round(y1 / height, 4),
                            round(x2 / width, 4),
                            round(y2 / height, 4),
                        ],
                        "confidence": round(float(c), 4),
                    }
                )
                unique_track_ids.add(int(tid))

        face_timeline.append(
            {
                "frame_index": len(face_timeline),
                "frame_original": frame_idx * interval,
                "timestamp_sec": round(timestamp, 2),
                "detections": detections,
            }
        )

        frame_idx += 1

    proc.stdout.close()
    proc.wait()

    print(
        f"[process] Done: {len(unique_track_ids)} tracks, "
        f"{len(face_timeline)} frames processed"
    )

    return {
        "faces": face_timeline,
        "unique_tracks": sorted(unique_track_ids),
        "fps": target_fps,
        "total_frames": len(face_timeline),
        "video": {
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
            "original_fps": round(orig_fps, 2),
        },
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job["input"]
    video_url = job_input.get("video_url")
    target_fps = job_input.get("fps", DEFAULT_FPS)
    conf = job_input.get("confidence", DEFAULT_CONF)
    iou = job_input.get("iou", DEFAULT_IOU)

    if not video_url:
        return {"error": "video_url is required"}

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video_path = tmp.name
    tmp.close()

    try:
        download_video(video_url, video_path)
        result = process_video(video_path, target_fps, conf, iou)
        print(
            f"[handler] Done: {len(result['unique_tracks'])} tracks, "
            f"{result['total_frames']} frames"
        )
        return result

    except Exception as e:
        print(f"[handler] Error: {e}")
        return {"error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
