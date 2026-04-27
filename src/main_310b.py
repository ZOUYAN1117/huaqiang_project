from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from utils.config import load_config
from camera.camera_reader import CameraReader


def main():
    cfg = load_config(PROJECT_ROOT / "configs" / "ascend.yaml")
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
    print("[INFO] Ascend pipeline started")

    while True:
        frame = camera.read()
        if frame is None:
            break

        # TODO: 后面接 OM 推理
        # outputs = ascend_engine.infer(frame)

    camera.release()


if __name__ == "__main__":
    main()