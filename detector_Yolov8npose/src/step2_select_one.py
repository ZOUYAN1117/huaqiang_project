from ultralytics import YOLO
import cv2


def select_main_person(persons):
    """
    从所有检测到的人中选一个主人物。
    当前策略：选面积最大的框。
    后续可替换为：
    - 离画面中心最近
    - 置信度最高
    - 跟踪ID优先
    """
    if not persons:
        return None
    return max(persons, key=lambda p: p["area"])


def main():
    # 加载模型
    model = YOLO("yolov8n.pt")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 推理
        results = model(frame, verbose=False)

        # 保存所有 person 检测结果
        persons = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # COCO 中 person 类别是 0
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # 边界保护，避免越界
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

        # 显示总人数（为后续多人分支留接口）
        cv2.putText(
            frame,
            f"persons: {len(persons)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        # 选主人物
        main_person = select_main_person(persons)

        if main_person is not None:
            x1, y1, x2, y2 = main_person["bbox"]
            conf = main_person["conf"]

            # 只给主人物画框
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

            # 裁剪主人物 ROI
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                cv2.imshow("Main Person ROI", roi)

        # 主画面显示
        cv2.imshow("YOLOv8n Main Person", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()