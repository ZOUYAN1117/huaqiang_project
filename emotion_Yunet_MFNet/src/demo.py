import cv2 as cv
import numpy as np
import datetime

from facial_fer_model import FacialExpressionRecog
from yunet import YuNet

# ===== 模型路径（用你当前目录的）=====
yunet_model_path = "face_detection_yunet_2023mar.onnx"
fer_model_path = "model.onnx"

# ===== 初始化模型 =====
detect_model = YuNet(modelPath=yunet_model_path)

fer_model = FacialExpressionRecog(
    modelPath=fer_model_path,
    backendId=cv.dnn.DNN_BACKEND_OPENCV,
    targetId=cv.dnn.DNN_TARGET_CPU
)

# ===== 可视化函数 =====
def visualize(image, det_res, fer_res):
    print('%s %3d faces detected.' % (datetime.datetime.now(), len(det_res)))

    output = image.copy()

    for ind, (det, fer_type) in enumerate(zip(det_res, fer_res)):
        bbox = det[0:4].astype(np.int32)

        fer_type_str = FacialExpressionRecog.getDesc(fer_type)

        print(f"Face {ind}: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {fer_type_str}")

        # 画框
        cv.rectangle(output,
                     (bbox[0], bbox[1]),
                     (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                     (0, 255, 0), 2)

        # 写表情
        cv.putText(output,
                   fer_type_str,
                   (bbox[0], bbox[1] - 5),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 0, 255),
                   2)

    return output


# ===== 推理函数 =====
def process(frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])

    dets = detect_model.infer(frame)

    if dets is None:
        return False, None, None

    fer_res = np.zeros(0, dtype=np.int8)

    for face_points in dets:
        fer_res = np.concatenate(
            (fer_res, fer_model.infer(frame, face_points[:-1])),
            axis=0
        )

    return True, dets, fer_res


# ===== 主函数（摄像头）=====
if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 摄像头打不开")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取失败")
            break

        status, dets, fer_res = process(frame)

        if status:
            frame = visualize(frame, dets, fer_res)

        cv.imshow("FER Demo", frame)

        # ESC退出
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()