from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMOTION_SRC = PROJECT_ROOT / "emotion_Yunet_MFNet" / "src"


def load_attr_from_file(module_name: str, file_path: Path, attr_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


FacialExpressionRecog = load_attr_from_file(
    "facial_fer_model",
    EMOTION_SRC / "facial_fer_model.py",
    "FacialExpressionRecog",
)
YuNet = load_attr_from_file("yunet", EMOTION_SRC / "yunet.py", "YuNet")


POSE_MODEL_PATH = PROJECT_ROOT / "models" / "onnx" / "yolov8n-pose.onnx"
YUNET_MODEL_PATH = PROJECT_ROOT / "models" / "onnx" / "face_detection_yunet_2023mar.onnx"
FER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "facial_expression_recognition_mobilefacenet_2022july_fixed.onnx"
)

POSE_INPUT_SIZE = 640
POSE_CONF_THRES = 0.25
POSE_IOU_THRES = 0.45
KEYPOINT_CONF_THRES = 0.30

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class PoseDetection:
    bbox: np.ndarray
    score: float
    keypoints: np.ndarray
    action: str


@dataclass
class FaceEmotion:
    bbox: np.ndarray
    emotion: str


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")


def letterbox(image: np.ndarray, size: int = POSE_INPUT_SIZE):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.zeros_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        order = order[np.where(iou <= iou_thres)[0] + 1]

    return keep


class YoloV8PoseDetector:
    def __init__(self, model_path: Path):
        require_file(model_path)
        self.net = cv.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def infer(self, frame: np.ndarray) -> list[PoseDetection]:
        input_image, scale, pad_x, pad_y = letterbox(frame)
        blob = cv.dnn.blobFromImage(
            input_image,
            scalefactor=1.0 / 255.0,
            size=(POSE_INPUT_SIZE, POSE_INPUT_SIZE),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()
        output = self._normalize_output(output)

        boxes_xywh = output[:, :4]
        scores = output[:, 4]
        keypoints = output[:, 5:].reshape(-1, 17, 3)

        valid = scores > POSE_CONF_THRES
        boxes_xywh = boxes_xywh[valid]
        scores = scores[valid]
        keypoints = keypoints[valid]

        if len(scores) == 0:
            return []

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        keep = nms(boxes_xyxy, scores, POSE_IOU_THRES)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        keypoints = keypoints[keep]

        boxes_xyxy, keypoints = self._restore_to_frame(
            boxes_xyxy, keypoints, scale, pad_x, pad_y, frame.shape
        )

        detections = []
        for box, score, kpts in zip(boxes_xyxy, scores, keypoints):
            detections.append(
                PoseDetection(
                    bbox=box,
                    score=float(score),
                    keypoints=kpts,
                    action=classify_action(kpts),
                )
            )
        return detections

    @staticmethod
    def _normalize_output(output: np.ndarray) -> np.ndarray:
        if output.ndim == 3 and output.shape[1] == 56:
            return output[0].transpose(1, 0)
        if output.ndim == 3 and output.shape[2] == 56:
            return output[0]
        if output.ndim == 2 and output.shape[1] == 56:
            return output
        raise ValueError(f"Unexpected YOLOv8-pose output shape: {output.shape}")

    @staticmethod
    def _restore_to_frame(
        boxes: np.ndarray,
        keypoints: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        frame_shape,
    ):
        h, w = frame_shape[:2]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale
        keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, w - 1)
        keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, h - 1)

        return boxes, keypoints


class EmotionDetector:
    def __init__(self, yunet_path: Path, fer_path: Path):
        require_file(yunet_path)
        require_file(fer_path)
        self.face_detector = YuNet(
            modelPath=str(yunet_path),
            confThreshold=0.6,
            nmsThreshold=0.3,
            topK=5000,
            backendId=cv.dnn.DNN_BACKEND_OPENCV,
            targetId=cv.dnn.DNN_TARGET_CPU,
        )
        self.fer_model = FacialExpressionRecog(
            modelPath=str(fer_path),
            backendId=cv.dnn.DNN_BACKEND_OPENCV,
            targetId=cv.dnn.DNN_TARGET_CPU,
        )

    def infer(self, frame: np.ndarray) -> list[FaceEmotion]:
        h, w = frame.shape[:2]
        self.face_detector.setInputSize([w, h])
        faces = self.face_detector.infer(frame)

        results = []
        for face in faces:
            # YuNet layout: x, y, w, h, five landmarks, score.
            if len(face) < 15:
                continue
            fer = self.fer_model.infer(frame, face[:-1])[0]
            results.append(
                FaceEmotion(
                    bbox=face[:4].astype(np.float32),
                    emotion=FacialExpressionRecog.getDesc(int(fer)),
                )
            )
        return results


def point_ok(keypoints: np.ndarray, index: int) -> bool:
    return keypoints[index, 2] >= KEYPOINT_CONF_THRES


def classify_action(keypoints: np.ndarray) -> str:
    left_wrist = 9
    right_wrist = 10
    left_shoulder = 5
    right_shoulder = 6
    left_elbow = 7
    right_elbow = 8
    nose = 0

    left_hand_up = (
        point_ok(keypoints, left_wrist)
        and point_ok(keypoints, left_shoulder)
        and keypoints[left_wrist, 1] < keypoints[left_shoulder, 1]
    )
    right_hand_up = (
        point_ok(keypoints, right_wrist)
        and point_ok(keypoints, right_shoulder)
        and keypoints[right_wrist, 1] < keypoints[right_shoulder, 1]
    )

    if left_hand_up and right_hand_up:
        return "cheer"
    if left_hand_up or right_hand_up:
        return "wave"

    if all(point_ok(keypoints, i) for i in [left_wrist, right_wrist, left_elbow, right_elbow, nose]):
        wrist_dist = np.linalg.norm(keypoints[left_wrist, :2] - keypoints[right_wrist, :2])
        elbow_dist = np.linalg.norm(keypoints[left_elbow, :2] - keypoints[right_elbow, :2])
        nose_y = keypoints[nose, 1]
        wrist_y = (keypoints[left_wrist, 1] + keypoints[right_wrist, 1]) / 2
        if wrist_dist < max(45.0, elbow_dist * 0.45) and wrist_y > nose_y:
            return "heart_like"

    return "normal"


def draw_pose(frame: np.ndarray, detections: list[PoseDetection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv.putText(
            frame,
            f"person {det.score:.2f} action:{det.action}",
            (x1, max(20, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
        )

        for idx, (x, y, conf) in enumerate(det.keypoints):
            if conf < KEYPOINT_CONF_THRES:
                continue
            cv.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv.putText(
                frame,
                str(idx),
                (int(x) + 3, int(y) + 3),
                cv.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 0, 0),
                1,
            )

        for a, b in SKELETON:
            if det.keypoints[a, 2] < KEYPOINT_CONF_THRES or det.keypoints[b, 2] < KEYPOINT_CONF_THRES:
                continue
            pt_a = (int(det.keypoints[a, 0]), int(det.keypoints[a, 1]))
            pt_b = (int(det.keypoints[b, 0]), int(det.keypoints[b, 1]))
            cv.line(frame, pt_a, pt_b, (255, 0, 0), 2)


def draw_emotions(frame: np.ndarray, emotions: list[FaceEmotion]) -> None:
    for item in emotions:
        x, y, w, h = item.bbox.astype(int)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv.putText(
            frame,
            f"emotion:{item.emotion}",
            (x, max(20, y - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )


def print_frame_event(pose_dets: list[PoseDetection], emotions: list[FaceEmotion]) -> None:
    main_action = pose_dets[0].action if pose_dets else "none"
    main_emotion = emotions[0].emotion if emotions else "none"
    print(
        {
            "persons": len(pose_dets),
            "faces": len(emotions),
            "main_action": main_action,
            "main_emotion": main_emotion,
        }
    )


def open_camera(camera_id: int, width: int, height: int, fps: int):
    cap = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera_id}")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    return cap


def main():
    pose_detector = YoloV8PoseDetector(POSE_MODEL_PATH)
    emotion_detector = EmotionDetector(YUNET_MODEL_PATH, FER_MODEL_PATH)

    camera_id = 1
    cap = open_camera(camera_id=camera_id, width=640, height=480, fps=30)

    print("[INFO] PC demo1 started")
    print("[INFO] Press 'q' or ESC to quit, press 'j' to print one frame event")

    last_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            break

        pose_dets = pose_detector.infer(frame)
        emotions = emotion_detector.infer(frame)

        draw_pose(frame, pose_dets)
        draw_emotions(frame, emotions)

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        cv.putText(
            frame,
            f"FPS:{fps:.1f} persons:{len(pose_dets)} faces:{len(emotions)}",
            (15, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        cv.imshow("Huaqiang PC Demo1 - Pose + Emotion", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("j"):
            print_frame_event(pose_dets, emotions)
        if key in (ord("q"), 27):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
