import cv2
from ultralytics import YOLO
import numpy as np
print(cv2.__version__)

print("OpenCV version:", cv2.__version__)

model = YOLO('C:/Users/johan/Documents/Projects/Python-Tasks/Day_19_YOLO/runs/detect/train6/weights/best.pt')  # load a pretrained model (recommended for training)

image_path = 'C:/Users/johan/Downloads/Yolo_Task/test/images/63228_78022002696_4063-00022.jpg'
img = cv2.imread(image_path)

results = model(
    image_path
)

x_coords = []

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        width = x2 - x1
        height = y2 - y1

        class_id =  int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"Object: {class_name}, Confidence: {conf:.2f}, Width: {width}, Height: {height}")

        x_coords.append(int((x1 + x2) / 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = f"{class_name} {conf:.2f}"

        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite(image_path, img)
cv2.waitKey()
cv2.destroyAllWindows()

        