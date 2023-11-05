from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.train(data="data.yaml", epochs=15, batch = 16,  name='Yolov8m')
