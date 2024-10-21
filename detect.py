from ultralytics import YOLO
 
# model = YOLO('/root/ultralytics6.1/runs/train/exp2/weights/best.pt')
model = YOLO('/root/ultralytics6.1/runs/train/exp4/weights/best.pt')

results = model('/root/ultralytics6.1/ultralytics/assets/img002769.jpg', save=True)