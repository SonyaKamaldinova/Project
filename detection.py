from ultralytics import YOLO
from PIL import Image
model = YOLO("/custom_yolov8/weights/best.pt")
results = model.predict("", show=True, save=True) #add path to image
result = results[0]
print(len(result.boxes))
for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = round(box.conf[0].item(),2)
    print("Object type:", label)
    print("Coordinates:", cords)
    print("Probablity:", prob)
    print("---")
Image.fromarray(result.plot()[:, :, ::-1])


