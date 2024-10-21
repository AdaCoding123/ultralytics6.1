import sys
sys.path.append("/root/ultralytics6.1/ultralytics6.1")
from ultralytics import YOLO
 

# model = YOLO("/root/ultralytics6.1/ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
model = YOLO("/root/ultralytics6.1/ultralytics/cfg/models/v8/yolov8.yaml").load("yolov8n.pt")  # 使用预训练权重训练


# Train the model
results = model.train(
    data="/root/ultralytics6.1/ultralytics/cfg/datasets/fireData.yaml", 
    epochs=200, 
    batch=16,
    imgsz=640,
    project='/root/ultralytics6.1/ultralytics6.1',
    name='exp',
    augment=True)


