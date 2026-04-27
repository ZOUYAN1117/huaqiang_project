import cv2
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMG_PATH = PROJECT_ROOT / "test_data" / "test.jpg"
META_PATH = PROJECT_ROOT / "ascend_outputs" / "preprocess_meta.txt"
OUTPUT_BIN_DIR = PROJECT_ROOT / "ascend_outputs"
SAVE_PATH = PROJECT_ROOT / "outputs" / "yolov8_pose_result.jpg"

CONF_THRES = 0.25
IOU_THRES = 0.45


SKELETON = [
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 6),                # shoulders
    (5, 11), (6, 12),      # body
    (11, 12),              # hips
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
]


def read_meta(path: Path) -> dict:
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split("=")
            meta[k] = float(v)
    return meta


def find_output_bin(output_dir: Path) -> Path:
    bins = list(output_dir.rglob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No .bin found in {output_dir}")

    # 一般最大那个就是 YOLO 输出
    bins = sorted(bins, key=lambda p: p.stat().st_size, reverse=True)
    print("[INFO] use output bin:", bins[0])
    print("[INFO] size:", bins[0].stat().st_size)
    return bins[0]


def load_yolov8_pose_output(bin_path: Path) -> np.ndarray:
    data = np.fromfile(str(bin_path), dtype=np.float32)

    print("[INFO] raw float count:", data.size)

    # YOLOv8n-pose 常见输出: 1 x 56 x 8400
    if data.size == 1 * 56 * 8400:
        output = data.reshape(1, 56, 8400)
        output = output[0].transpose(1, 0)  # 8400 x 56
    elif data.size == 1 * 8400 * 56:
        output = data.reshape(1, 8400, 56)[0]
    else:
        raise ValueError(
            f"Unexpected output size: {data.size}. "
            "Expected 470400 = 1*56*8400."
        )

    print("[INFO] parsed output shape:", output.shape)
    return output


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float):
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
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def restore_boxes_and_keypoints(boxes, keypoints, meta, img_shape):
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]

    h, w = img_shape[:2]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale
    keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale

    keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, w - 1)
    keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, h - 1)

    return boxes, keypoints


def draw_pose(img, boxes, scores, keypoints):
    for box, score, kpts in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"person {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for i, (x, y, conf) in enumerate(kpts):
            if conf < 0.3:
                continue
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(
                img,
                str(i),
                (int(x) + 3, int(y) + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1,
            )

        for a, b in SKELETON:
            if kpts[a, 2] < 0.3 or kpts[b, 2] < 0.3:
                continue

            xa, ya = int(kpts[a, 0]), int(kpts[a, 1])
            xb, yb = int(kpts[b, 0]), int(kpts[b, 1])
            cv2.line(img, (xa, ya), (xb, yb), (255, 0, 0), 2)

    return img


def main():
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise FileNotFoundError(IMG_PATH)

    meta = read_meta(META_PATH)
    bin_path = find_output_bin(OUTPUT_BIN_DIR)

    output = load_yolov8_pose_output(bin_path)

    boxes_xywh = output[:, 0:4]
    scores = output[:, 4]

    kpts_raw = output[:, 5:]  # 17 * 3 = 51
    keypoints = kpts_raw.reshape(-1, 17, 3)

    valid = scores > CONF_THRES

    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    keypoints = keypoints[valid]

    print("[INFO] candidates after conf:", len(scores))

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    keep = nms(boxes_xyxy, scores, IOU_THRES)

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    keypoints = keypoints[keep]

    print("[INFO] persons after nms:", len(scores))

    boxes_xyxy, keypoints = restore_boxes_and_keypoints(
        boxes_xyxy,
        keypoints,
        meta,
        img.shape,
    )

    result = draw_pose(img.copy(), boxes_xyxy, scores, keypoints)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SAVE_PATH), result)

    print("[OK] saved result:", SAVE_PATH)

    cv2.imshow("YOLOv8 Pose Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()