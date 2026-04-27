from pathlib import Path
import os

# =============================
# 0. 路径与缓存目录（必须放在 tensorflow_hub 导入前）
# =============================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CACHE_DIR = PROJECT_ROOT / "cache"
TFHUB_CACHE_DIR = CACHE_DIR / "tfhub"

TFHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = str(TFHUB_CACHE_DIR)

print(f"[INFO] PROJECT_ROOT     = {PROJECT_ROOT}")
print(f"[INFO] TFHUB_CACHE_DIR  = {TFHUB_CACHE_DIR}")

# =============================
# 1. 其余导入
# =============================
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import time

# =============================
# 2. 配置
# =============================
YOLO_MODEL_PATH = SRC_DIR / "yolov8n.pt"
MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
INPUT_SIZE = 192  # MoveNet Lightning 输入尺寸

KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def select_main_person(persons):
    if not persons:
        return None
    return max(persons, key=lambda p: p["area"])


def load_movenet(model_url):
    print("[INFO] 开始解析 MoveNet 路径 ...")
    resolved_path = hub.resolve(model_url)
    print(f"[INFO] MoveNet 缓存路径 = {resolved_path}")

    print("[INFO] 开始加载 MoveNet ...")
    model = hub.load(model_url)
    print("[INFO] MoveNet 加载完成")
    return model.signatures["serving_default"]


def run_movenet(movenet, roi_bgr):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(roi_rgb, (INPUT_SIZE, INPUT_SIZE))
    input_image = np.expand_dims(resized, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    outputs = movenet(input_image)
    keypoints = outputs["output_0"].numpy()[0, 0, :, :]  # [17, 3]
    return keypoints


def draw_keypoints_on_roi(roi, keypoints, score_th=0.2):
    h, w, _ = roi.shape
    pts = []

    for kp in keypoints:
        y, x, score = kp
        px = int(x * w)
        py = int(y * h)
        pts.append((px, py, float(score)))

    for i, j in EDGES:
        x1, y1, s1 = pts[i]
        x2, y2, s2 = pts[j]
        if s1 > score_th and s2 > score_th:
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for x, y, score in pts:
        if score > score_th:
            cv2.circle(roi, (x, y), 4, (0, 255, 255), -1)

    return pts


def build_json_result(main_person, keypoints, roi_shape, person_count):
    h, w = roi_shape[:2]
    keypoint_list = []

    for name, kp in zip(KEYPOINT_NAMES, keypoints):
        y, x, score = kp
        keypoint_list.append({
            "name": name,
            "x": float(x * w),
            "y": float(y * h),
            "score": float(score)
        })

    result = {
        "timestamp": time.time(),
        "main_person": {
            "bbox": main_person["bbox"],
            "conf": float(main_person["conf"]),
            "pose": {
                "keypoints": keypoint_list
            }
        },
        "multi_person_stub": {
            "count": int(person_count),
            "reserved": True
        }
    }
    return result


def main():
    if not YOLO_MODEL_PATH.exists():
        print(f"[ERROR] 未找到 YOLO 权重文件: {YOLO_MODEL_PATH}")
        return

    print(f"[INFO] 加载 YOLOv8n: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    print("[INFO] 加载 MoveNet Lightning")
    try:
        movenet = load_movenet(MOVENET_MODEL_URL)
    except Exception as e:
        print("[ERROR] MoveNet 加载失败：", e)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    last_json = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取摄像头画面")
            break

        results = yolo_model(frame, verbose=False)
        persons = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    w = x2 - x1
                    h = y2 - y1
                    area = w * h

                    if w > 0 and h > 0:
                        persons.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                            "area": area
                        })

        cv2.putText(
            frame,
            f"persons: {len(persons)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        main_person = select_main_person(persons)

        if main_person is not None:
            x1, y1, x2, y2 = main_person["bbox"]
            conf = main_person["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"main {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            roi = frame[y1:y2, x1:x2].copy()

            if roi.size > 0:
                keypoints = run_movenet(movenet, roi)
                draw_keypoints_on_roi(roi, keypoints, score_th=0.2)

                last_json = build_json_result(main_person, keypoints, roi.shape, len(persons))

                valid_points = sum(1 for kp in keypoints if kp[2] > 0.2)
                cv2.putText(
                    roi,
                    f"valid_kpts: {valid_points}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Main Person ROI + MoveNet", roi)

        cv2.imshow("YOLOv8n Main Person", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("j"):
            if last_json is not None:
                print(json.dumps(last_json, ensure_ascii=False, indent=2))
        elif key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()