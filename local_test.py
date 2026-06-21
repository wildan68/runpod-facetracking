"""
Local test — simulates RunPod's handler() call.
Usage:
    python local_test.py <video_url>
Example:
    python local_test.py https://storage.klip.yt/some-video.mp4
"""
import sys
import json
from handler import handler

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("video_url: ")

    mock_job = {
        "input": {
            "video_url": url,
            "fps": 2.0,
            "confidence": 0.4,
            "iou": 0.5,
        }
    }

    result = handler(mock_job)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))

    if result.get("error"):
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print(f"\nTracks detected: {len(result.get('unique_tracks', []))}")
    print(f"Frames processed: {result.get('total_frames', 0)}")
