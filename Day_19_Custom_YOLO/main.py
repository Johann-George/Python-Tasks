import cv2
import json
import pandas as pd
import os
import json
from yolo import perform_yolo_algo
import re

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

img_file_path = "/home/johannvgeorge/Downloads/dataset 1/dataset/63228_78022002696_4063-00063.jpg"
json_data = "/home/johannvgeorge/Downloads/dataset 1/dataset/63228_78022002696_4063-00063.json"
with open(json_data, 'r') as file:
    data_var = json.load(file)
draw_boxes_from_json(img_file_path, data_var)


def load_ocr_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    annotations = data["textAnnotations"][1:]  # skip index 0 (full text)
    extracted = []

    for ann in annotations:
        desc = ann["description"]
        vertices = ann["boundingPoly"]["vertices"]

        # get left-top corner
        x1 = vertices[0].get("x", 0)
        y1 = vertices[0].get("y", 0)
        x2 = vertices[1].get("x", 0)
        y2 = vertices[1].get("y", 0)

        extracted.append((desc, x1, y1, x2, y2))
    
    return extracted


def preprocess_x_coords(x_coords):
    x_coords = sorted(list(set(x_coords)))
    x_coords.append(99999)  # large number as last boundary
    return x_coords

def assign_to_columns(ocr_data, x_coords):
    table = []

    for text, x1, y1, x2, y2 in ocr_data:
        col_index = None
        
        x_mid = (x1 + x2)/2
        # determine column by x boundaries
        for i in range(len(x_coords) - 1):
            if x_coords[i] <= x2 < x_coords[i+1]:
                col_index = i
                break
        
        table.append({
            "text": text,
            "x": x1,
            "y": y1,
            "column": col_index
        })
    
    return table

def group_rows(table, threshold=25):
    table_sorted = sorted(table, key=lambda t: t["y"])

    rows = []
    current_row = []
    last_y = None
    i = 0
    for item in table_sorted:
        if i < 12:
            i+=1
            continue
        if last_y is None:
            current_row.append(item)
        elif item["text"] == "..." or item["text"] == ".":
            continue
        elif abs(item["y"] - last_y) <= threshold:
            if item["text"].isdigit() and len(item["text"]) == 4:
                continue
            # item["text"] = re.sub(r'\d\d\d\d', '', item["text"])
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]

        last_y = item["y"]

    if current_row:
        rows.append(current_row)

    return rows

def build_table(rows, num_columns):
    final_table = []

    for row in rows:
        row_data = [""] * num_columns
        
        row = sorted(row, key=lambda t: t["x"])

        for item in row:
            if item["column"] is not None:
                cleaned = clean_text(item["text"])
                if cleaned:  # only append non-empty results
                    row_data[item["column"]] += cleaned + " "

        row_data = [col.strip() for col in row_data]
        final_table.append(row_data)

    return final_table

def clean_text(text):
    if text is None:
        return ""

    # Remove punctuations you don't want
    text = re.sub(r'[./°\\•]', '', text)

    # Remove leading 4-digit numbers
    text = re.sub(r'^\d{4}\s*', '', text)

    # Remove leading 3-digit numbers
    text = re.sub(r'^\d{3}\s*', '', text)

    # Remove isolated 4-digit numbers anywhere
    text = re.sub(r'\b\d{4}\b', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def generate_table(json_path, x_coords):
    # Step 1: Load OCR JSON
    ocr_data = load_ocr_data(json_path)

    print("Before preprocessing:", x_coords)
    # Step 2: Process column boundaries
    x_coords = preprocess_x_coords(x_coords)
    print("After preprocessing:", x_coords)
    num_columns = len(x_coords) - 1

    # Step 3: Assign annotations to columns
    table_assignments = assign_to_columns(ocr_data, x_coords)

    # Step 4: Group into rows
    row_groups = group_rows(table_assignments)

    # Step 5: Create final table
    final_table = build_table(row_groups, num_columns)

    return final_table


#not used
def process_text_annotations(json_data):
    """
    Extracts description and top-left coordinates from textAnnotations.
    """        

    processed_data = [] 
    for annotation in json_data.get('textAnnotations', []):
        description = annotation.get('description', '').replace('\n', ' ')
        vertices = annotation.get('boundingPoly', {}).get('vertices', [])
        if vertices:
            top_left_x = vertices[0].get('x', 0)
            top_left_y = vertices[0].get('y', 0)
        else:
            top_left_x = 0
            top_left_y = 0

        processed_data.append({
            'Description': description,
            'TopLeftX': top_left_x,
            'TopLeftY': top_left_y
        })
    return processed_data

#Not used
def organize_by_coordinates(processed_data, y_tolerance=10, x_tolerance=200):
    processed_data.sort(key=lambda item: (item['TopLeftY'], item['TopLeftX']))

    organized_rows = []
    current_row = []

    for item in processed_data:
        if not current_row:
            current_row.append(item)
            continue

        if abs(item['TopLeftY'] - current_row[0]['TopLeftY']) <= y_tolerance:
            current_row.append(item)
        else:
            organized_rows.append([d['Description'] for d in current_row])
            current_row = [item]
    if current_row:
        organized_rows.append([d['Description'] for d in current_row])

    return processed_data

def save_to_excel(data, output_filename='output.xlsx'):
    """
    Converts a list of dictionaries to pandas dataframe and saves to Excel.
    """
    df = pd.DataFrame(data)
    df.to_excel(output_filename, index=False, engine='openpyxl')
    print(f"Data successfully saved to {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    # with open('C:/Users/johan/Downloads/dataset_1/dataset/63228_78022002696_4063-00063.json') as file:
    #     json_data = json.load(file)
    # extracted_data = process_text_annotations(json_data)
    # structured_data = organize_by_coordinates(extracted_data)
    # save_to_excel(structured_data, 'output_with_coordinates.xlsx')
    json_path = '/home/johannvgeorge/Downloads/dataset 1/dataset/63228_78022002696_4063-00063.json'
    x_coords, full_address_x_coords = perform_yolo_algo()
    df = pd.DataFrame(generate_table(json_path, x_coords))
    df.to_excel("output.xlsx", index=True)