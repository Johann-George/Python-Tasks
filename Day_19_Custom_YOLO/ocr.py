import cv2
import json
import numpy as np

def draw_boxes_from_json(image_path, json_data):
    img = cv2.imread(image_path)
    annotations = json_data.get("textAnnotations", [])

    for annotation in annotations:
        if "boundingPoly" in annotation and "vertices" in annotation["boundingPoly"]:
            vertices = annotation["boundingPoly"]["vertices"]
            if len(vertices) == 4:
                x_coords = [v.get("x", 0) for v in vertices]
                y_coords = [v.get("y", 0) for v in vertices]

                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)

                label = annotation.get("description", "")

                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    text_color = (0, 0, 255)
                    cv2.putText(img, label, (xmin, ymin - 10), font, font_scale, text_color, thickness)

    cv2.imwrite("bb.jpg", img)
    # cv2.imshow("Image with bounding boxes:", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_file_path = "C:/Users/johan/Downloads/dataset_1/dataset/63228_78022002696_4063-00063.jpg"
json_data = "C:/Users/johan/Downloads/dataset_1/dataset/63228_78022002696_4063-00063.json"
with open(json_data, 'r') as file:
    data_var = json.load(file)
draw_boxes_from_json(img_file_path, data_var)
        

def save_to_excel():
