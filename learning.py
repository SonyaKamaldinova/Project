from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data="data.yaml", epochs=30,  name='Yolov8n')
