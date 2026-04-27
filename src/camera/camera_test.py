import cv2
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from utils.config import load_config
from camera.camera_reader import CameraReader


def main():
    config_path = PROJECT_ROOT / "configs" / "pc.yaml"
    cfg = load_config(config_path)

    cam_cfg = cfg["camera"]

    camera = CameraReader(
        source_type=cam_cfg.get("source_type", "camera"),
        camera_id=cam_cfg.get("camera_id", 0),
        video_path=cam_cfg.get("video_path", None),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
    )

    camera.open()

    last_time = time.time()
    frame_count = 0

    while True:
        frame = camera.read()
        if frame is None:
            print("[WARN] No frame")
            break

        frame_count += 1
        now = time.time()

        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now
            print(f"[INFO] FPS: {fps:.2f}")

        cv2.imshow("camera_test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()