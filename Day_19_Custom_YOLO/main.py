import cv2
import json
import pandas as pd
import os
import json
from yolo import perform_yolo_algo
import re
from openpyxl import load_workbook

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
        
        # x_mid = (x1 + x2)/2
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
    full_address = []
    current_row = []
    last_y = None
    i = 0
    s = ""
    for item in table_sorted:
        if i < 35:
            i+=1
            continue
        if last_y is None:
            current_row.append(item)
        elif item["text"] == "..." or item["text"] == "." or item["text"] == "..":
            continue
        elif abs(item["y"] - last_y) <= threshold:
            if item["text"].isdigit() and len(item["text"]) == 4:
                continue
            current_row.append(item)
        else:
            print("Current Row=",current_row)
            #     if item1["text"].isupper():
            #         if s != "":
            #             s += " " + item1["text"]
            #         else:
            #             s = item1["text"]
            #         count_address = 1
            #         break
            # if count_address == 0:
            #     full_address.append(s)
            #     rows.append(current_row)
            # current_row = [item]
            count_address = 0
            for item1 in current_row:
                if item1["text"] != '':
                    if not is_pascal_case(item1["text"]):
                        count_address += 1
            if count_address == 1 or count_address == 2 or count_address == 3:
                s = ""
                for item2 in current_row:
                    if s != "":
                        s+=" "
                    s += item2["text"] 
                    current_row = [item]
            else:
                full_address.append(s)
                rows.append(current_row)
                current_row = [item]

        last_y = item["y"]

    if current_row:
        rows.append(current_row)
        full_address.append(s)

    return rows, full_address

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
    row_groups, full_address = group_rows(table_assignments)

    # Step 5: Create final table
    final_table = build_table(row_groups, num_columns)

    return final_table, full_address

def save_to_excel(data, output_filename='output.xlsx'):
    """
    Converts a list of dictionaries to pandas dataframe and saves to Excel.
    """
    df = pd.DataFrame(data)
    df.to_excel(output_filename, index=False, engine='openpyxl')
    print(f"Data successfully saved to {os.path.abspath(output_filename)}")

def is_pascal_case_advanced(s):
    if not s:
        return False
    
    # Check for traditional PascalCase (e.g., 'MyVariableName')
    # Starts with an uppercase letter, followed by any number of letters/digits,
    # then optionally followed by more words starting with uppercase letters.
    if re.fullmatch(r'([A-Z][a-zA-Z0-9]*)+', s):
        return True

    # Check for PascalCase-like phrases with spaces (e.g., 'John Street')
    # Each word starts with an uppercase letter.
    words = s.split()
    if words and all(word and word[0].isupper() for word in words):
        return True

    return False

def is_pascal_case(s):
    # Check if the string is non-empty, starts with an uppercase letter, 
    # and contains only letters and numbers (no spaces/underscores)
    return bool(s) and s[0].isupper() and s.isalnum() and not re.search(r'[a-z][A-Z]', s) is None

if __name__ == "__main__":
    # with open('C:/Users/johan/Downloads/dataset_1/dataset/63228_78022002696_4063-00063.json') as file:
    #     json_data = json.load(file)
    # extracted_data = process_text_annotations(json_data)
    # structured_data = organize_by_coordinates(extracted_data)
    # save_to_excel(structured_data, 'output_with_coordinates.xlsx')
    json_path = '/home/johannvgeorge/Downloads/dataset 1/dataset/63228_78022002696_4063-00063.json'
    x_coords, full_address_x_coords = perform_yolo_algo()
    table, full_address = generate_table(json_path, x_coords)
    print("Table=", table)
    df = pd.DataFrame(table, columns=['Names of Electors in full, Surname being first', 
                                      'Place of Abode',
                                      'Nature of Qualification',
                                      'Description of Qualifying Property',
                                    ])
    column_to_split = df.iloc[:, 0]
    print(column_to_split)
    new_columns = column_to_split.str.split(',', expand=True)
    df['First Name'] = new_columns[0]
    df['Last Name'] = new_columns[1]
    df['Sub Address'] = full_address
    # updated_full_address = [word for word in full_address if any(char.isupper() for char in word)]
    # updated_full_address = [item for item in full_address if is_pascal_case_advanced(item)]
    # updated_full_address = [item for item in updated_full_address if not ' ' not in item]
    # for i in range(len(full_address)-1, -1, -1):
    #     word = full_address[i]
    #     if word.islower() or ' ' not in word:
    #         del full_address[i]
    # print("Updated Full Address=",updated_full_address)
    # df['Full Address'] = pd.Series(updated_full_address)
    # df['Full Address'] = full_address
    df.to_excel("output.xlsx", index=True)
    