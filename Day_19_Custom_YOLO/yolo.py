import cv2
from ultralytics import YOLO
import numpy as np
print(cv2.__version__)
import torch
print(torch.version.cuda)
print("OpenCV version:", cv2.__version__)

image_path = '/home/johannvgeorge/Downloads/dataset 1/dataset/63228_78022002696_4063-00063.jpg'

def perform_yolo_algo():

    model = YOLO('/home/johannvgeorge/Documents/projects/Python_Tasks/Day_19_YOLO/runs/detect/train6/weights/best.pt')  # load a pretrained model (recommended for training)
    img = cv2.imread(image_path)

    results = model(
        image_path
    )

    x_coords = []
    full_add_x_coords = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            width = x2 - x1
            height = y2 - y1

            class_id =  int(box.cls[0])
            conf = float(box.conf[0])

            print("Entered here")
            class_name = model.names[class_id]
            if class_name == "Full Address":
                full_add_x_coords.append(x1)
            else:
                x_coords.append(x1)

            print(f"Object: {class_name}, Confidence: {conf:.2f}, Width: {width}, Height: {height}, X1:{x1}, X2:{x2}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{class_name} {conf:.2f}"

            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite('yolo.jpg', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return x_coords, full_add_x_coords

        