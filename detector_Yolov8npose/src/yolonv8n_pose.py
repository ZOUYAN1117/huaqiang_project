from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

results = model.predict(
    source="test.jpg",
    save=True,
    conf=0.25
)

for result in results:
    if result.keypoints is None:
        print("没有检测到人体关键点")
        continue

    # xy: 每个人体的关键点坐标，形状大致为 [人数, 关键点数, 2]
    keypoints_xy = result.keypoints.xy

    # conf: 每个关键点的置信度
    keypoints_conf = result.keypoints.conf

    print("关键点坐标：")
    print(keypoints_xy)

    print("关键点置信度：")
    print(keypoints_conf)