import cv2
import pytesseract
import os
import pandas as pd
import numpy as np
from main import ImagePreprocessor , detect_vertical_lines

class OCRExtraction:
    # Add this list of column boundaries to exclude
    EXCLUDED_COLUMNS = [253, 515, 570, 915, 964]

    def __init__(self, processed_img, original_img, column_boundaries=None):
        self.gray = processed_img
        self.img = original_img
        self.column_boundaries = column_boundaries or []
        self.line_groups = {}
        self.final_text = ""
        self.data_by_rows = []

    def extract_text(self, line_mid_variance=10):
        data = pytesseract.image_to_data(self.gray, output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if not word:
                continue
            x, y, w, h, conf = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i],
                int(data['conf'][i]),
            )
            if conf < 0:
                continue
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            y_mid = y + h // 2
            matched_line = None
            for line_y in self.line_groups.keys():
                if abs(line_y - y_mid) <= line_mid_variance:
                    matched_line = line_y
                    break
            if matched_line is not None:
                self.line_groups[matched_line].append((x, word))
            else:
                self.line_groups[y_mid] = [(x, word)]

    def isolate_vertical_lines(self, binarized=None):
        if binarized is None:
            binarized = self.gray
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(20, binarized.shape[0] // 20))
        )
        temp_img = cv2.erode(binarized, vertical_kernel, iterations=1)
        vertical_lines_img = cv2.dilate(temp_img, vertical_kernel, iterations=2)
        return vertical_lines_img

    def draw_vertical_lines(
        self, color=(255, 0, 0), thickness=2, draw_isolated=True, min_line_height=1000
    ):
        # Filter boundaries before drawing
        filtered_boundaries = [
            x for x in self.column_boundaries if x not in self.EXCLUDED_COLUMNS
        ]
        if filtered_boundaries:
            h = self.img.shape[0]
            for x in filtered_boundaries:
                cv2.line(self.img, (x, 0), (x, h), color, thickness)
        if draw_isolated:
            isolated_lines_img = self.isolate_vertical_lines()
            contours, _ = cv2.findContours(
                isolated_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h >= min_line_height and w < 8:
                    cv2.line(self.img, (x + w // 2, y), (x + w // 2, y + h), (0, 0, 255), 1)

    def reconstruct_text(self):
        sorted_lines = sorted(self.line_groups.items(), key=lambda kv: kv[0])
        final_text_lines = []
        table_started = False
        for _, words in sorted_lines:
            words = sorted(words, key=lambda w: w[0])
            line_text = " ".join([w[1] for w in words])
            if not table_started and "TAMES OF" in line_text.upper():
                table_started = True
            if table_started:
                final_text_lines.append(line_text)
        self.final_text = "\n".join(final_text_lines)
        return self.final_text

    def group_into_columns(self):
        # Filter unwanted boundaries before splitting table
        filtered_boundaries = [
            x for x in self.column_boundaries if x not in self.EXCLUDED_COLUMNS
        ]
        if not filtered_boundaries:
            print("[WARN] No column boundaries detected — skipping table grouping.")
            return None
        sorted_lines = sorted(self.line_groups.items(), key=lambda kv: kv[0])
        table_started = False
        temp_rows = []
        for _, words in sorted_lines:
            row = [""] * (len(filtered_boundaries) + 1)
            words = sorted(words, key=lambda w: w[0])
            line_text = " ".join([w[1] for w in words])
            if not table_started and "TAMES OF" in line_text.upper():
                table_started = True
            if table_started:
                for x, word in words:
                    col_index = self.get_column_index(x, filtered_boundaries)
                    row[col_index] += (word + " ")
                row = [cell.strip() for cell in row]
                temp_rows.append(row)
        # Extra: Truncate to first "TAMES OF" just in case
        for idx, row in enumerate(temp_rows):
            joined = " ".join(row).upper()
            if "TAMES OF" in joined:
                self.data_by_rows = temp_rows[idx:]
                break
        else:
            self.data_by_rows = temp_rows
        return self.data_by_rows

    def get_column_index(self, x, boundaries=None):
        boundaries = boundaries if boundaries is not None else [
            c for c in self.column_boundaries if c not in self.EXCLUDED_COLUMNS
        ]
        for i in range(len(boundaries)):
            if x < boundaries[i]:
                return i
        return len(boundaries)

    def save_results(self, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        text_path = os.path.join(output_dir, "output_text_path.jpg")
        bbox_image_path = os.path.join(output_dir, "output_psm_6_oem_1_contrast.jpg")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(self.final_text)
        cv2.imwrite(bbox_image_path, self.img)
        print(f"[INFO] Text saved at: {text_path}")
        print(f"[INFO] Image with bounding boxes saved at: {bbox_image_path}")

    def save_table_to_excel(self, output_path="outputs/table_extracted.xlsx"):
        if not self.data_by_rows:
            print("[WARN] No table data available to save.")
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(self.data_by_rows)
        df = pd.DataFrame(self.data_by_rows)
        df.to_excel(output_path, index=False, header=False)
        print(f"[INFO] Table data saved to: {output_path}")
 


ocr = OCRExtraction("img.jpg", "preprocessed_census.jpg",detect_vertical_lines(image_path="img.jpg") )


import cv2
import pytesseract
from image_preprocessor import ImagePreprocessor
from ocr_extractor import OCRExtraction


class OCR:
    def __init__(self, image_path, output_dir="outputs"):
        self.image_path = image_path
        self.output_dir = output_dir

    def run(self):
        # Step 1: Preprocess image
        preprocessor = ImagePreprocessor(self.image_path)
        preprocessor.preprocess()

        # Use the attributes now set by preprocess()
        gray = preprocessor.processed_image
        img = preprocessor.original_image

        # Step 2: Detect vertical lines (for table columns)
        column_boundaries = detect_vertical_lines()
        print(f"Detected column boundaries: {column_boundaries}")

        # Step 3: OCR text extraction
        extractor = OCRExtraction(gray, img, column_boundaries)
        extractor.extract_text()
        extractor.draw_vertical_lines()

        # Step 4: Reconstruct and save plain text
        final_text = extractor.reconstruct_text()
        extractor.save_results(self.output_dir)

        # Step 5: Group text into table columns and save to Excel
        extractor.group_into_columns()
        extractor.save_table_to_excel(f"{self.output_dir}/table_extracted.xlsx")

        # Step 6: Display output
        cv2.imshow("Processed Image", gray)
        cv2.imshow("Detected Text Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pipeline = OCR("test_image.jpg")
    pipeline.run()
 