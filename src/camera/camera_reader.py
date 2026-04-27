import cv2
from pathlib import Path


class CameraReader:
    def __init__(
        self,
        source_type: str = "camera",
        camera_id: int = 0,
        video_path: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.source_type = source_type
        self.camera_id = camera_id
        self.video_path = video_path
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        if self.source_type == "camera":
            self.cap = cv2.VideoCapture(self.camera_id)
        elif self.source_type == "video":
            if self.video_path is None:
                raise ValueError("video_path is required when source_type='video'")
            if not Path(self.video_path).exists():
                raise FileNotFoundError(self.video_path)
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera/video source")

        if self.source_type == "camera":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[INFO] Camera opened: {real_w}x{real_h}, fps={real_fps}")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() first.")

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None