import os
import cv2
import pytesseract
import numpy as np
import pandas as pd

# -------------------------------------------
# Preprocessing (integrated)
# -------------------------------------------

def preprocess_image(image_path, method='advanced', save_steps=False):
    """
    Returns: (original_bgr, processed_grayscale_or_binary, scale_factor)
    method ∈ {'original','grayscale','contrast','bilateral','advanced','otsu','adaptive'}
    scale_factor: ratio of processed image size to original (for coordinate conversion)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"cv2.imread failed for: {image_path}")

    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def clahe(gray, clip=3.0, tile=(8, 8)):
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        return c.apply(gray)

    def upscale(gray, scale=1.5):
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def denoise(gray):
        return cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)

    def binarize_otsu(gray):
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def binarize_adaptive(gray):
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            31, 8
        )

    scale_factor = 1.0  # Track the scale factor for coordinate conversion

    if method == 'original':
        processed = to_gray(img_bgr)

    elif method == 'grayscale':
        processed = to_gray(img_bgr)

    elif method == 'contrast':
        gray = to_gray(img_bgr)
        processed = clahe(gray)
        if save_steps:
            cv2.imwrite('step_contrast_gray.jpg', gray)
            cv2.imwrite('step_contrast_clahe.jpg', processed)

    elif method == 'bilateral':
        gray = to_gray(img_bgr)
        blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        processed = binarize_adaptive(blur)

    elif method == 'advanced':
        gray = to_gray(img_bgr)
        scale_factor = 1.7  # Track the upscale factor
        up = upscale(gray, scale_factor)
        de = denoise(up)
        cl = clahe(de)
        th = binarize_otsu(cl)
        processed = th
        if save_steps:
            cv2.imwrite('step_adv_upscaled.jpg', up)
            cv2.imwrite('step_adv_denoised.jpg', de)
            cv2.imwrite('step_adv_clahe.jpg', cl)
            cv2.imwrite('step_adv_otsu.jpg', th)

    elif method == 'otsu':
        gray = to_gray(img_bgr)
        processed = binarize_otsu(gray)

    elif method == 'adaptive':
        gray = to_gray(img_bgr)
        processed = binarize_adaptive(gray)

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    return img_bgr, processed, scale_factor


# -------------------------------------------
# Column detection (integrated)
# -------------------------------------------

def detect_vertical_lines(image_path='img.jpg', binarized=None, min_line_height_ratio=0.6):
    output_path = "detected_lines.jpg"

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    vertical_lines = []
    min_length = 100
    x_merge_tolerance = 15

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if abs(angle) > 85 and length >= min_length:
                vertical_lines.append((x1, y1, x2, y2, length))

    vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

    merged_lines = []
    if vertical_lines:
        current_group = [vertical_lines[0]]

        for line in vertical_lines[1:]:
            prev_x = np.mean([current_group[-1][0], current_group[-1][2]])
            curr_x = np.mean([line[0], line[2]])

            if abs(curr_x - prev_x) < x_merge_tolerance:
                current_group.append(line)
            else:
                merged_lines.append(max(current_group, key=lambda l: l[4]))
                current_group = [line]

        merged_lines.append(max(current_group, key=lambda l: l[4]))

    display_lines = [line for line in merged_lines if line[4] > 500]

    for x1, y1, x2, y2, length in display_lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(img, f"{int(length)}px", (x1 + 5, min(y1, y2) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"✅ Output saved to: {output_path}")
 
    x_coords = sorted([int((x1 + x2) / 2) for x1, y1, x2, y2, _ in display_lines])
    print("\nFiltered vertical line x-coordinates:", x_coords)

    return x_coords 


# -------------------------------------------
# OCR Extraction (Enhanced with row-wise extraction)
# -------------------------------------------

class OCRExtraction:
    EXCLUDED_COLUMNS = [175, 395, 434]  # Exclude unnecessary vertical lines

    def __init__(self, processed_img, original_img, column_boundaries=None, scale_factor=1.0):
        self.gray = processed_img
        self.img = original_img
        self.column_boundaries = column_boundaries or []
        self.scale_factor = scale_factor  # Scale factor for coordinate conversion
        self.line_groups = {}
        self.final_text = ""
        self.data_by_rows = []
        self.raw_rows = []  # Store raw row data before table structuring
        self.column_names = ['No', 'Name of children', 'Age', 'Address', 'Name of guardian or parent']
        
        # Scale column boundaries to match the processed image coordinates
        self.scaled_boundaries = [int(x * scale_factor) for x in self.column_boundaries]
        print(f"[INFO] Scale factor: {scale_factor}")
        print(f"[INFO] Original boundaries: {self.column_boundaries}")
        print(f"[INFO] Scaled boundaries: {self.scaled_boundaries}")

    def extract_text(self, line_mid_variance=10, config='--psm 6 --oem 1', min_conf=0):
        """Extract text and group by rows based on vertical position"""
        data = pytesseract.image_to_data(self.gray, output_type=pytesseract.Output.DICT, config=config)
        
        for i in range(len(data['text'])):
            word = (data['text'][i] or "").strip()
            if not word:
                continue
            try:
                conf = int(float(data['conf'][i]))
            except:
                conf = -1
            if conf < min_conf:
                continue

            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            y_mid = y + h // 2

            # Group by line midpoint
            matched_line = None
            for line_y in list(self.line_groups.keys()):
                if abs(line_y - y_mid) <= line_mid_variance:
                    matched_line = line_y
                    break

            if matched_line is not None:
                self.line_groups[matched_line].append((x, word, conf))
            else:
                self.line_groups[y_mid] = [(x, word, conf)]

    def extract_rows(self):
        """Extract text row by row, sorted by vertical position"""
        sorted_lines = sorted(self.line_groups.items(), key=lambda kv: kv[0])
        
        self.raw_rows = []
        for y_pos, words in sorted_lines:
            # Sort words in each row by x position (left to right)
            words_sorted = sorted(words, key=lambda w: w[0])
            
            row_data = {
                'y_position': y_pos,
                'words': [(x, word, conf) for x, word, conf in words_sorted],
                'full_text': ' '.join([word for _, word, _ in words_sorted]),
                'avg_confidence': np.mean([conf for _, _, conf in words_sorted]) if words_sorted else 0
            }
            self.raw_rows.append(row_data)
        
        print(f"\n{'='*80}")
        print(f"ROW-WISE TEXT EXTRACTION")
        print(f"{'='*80}")
        print(f"Total rows detected: {len(self.raw_rows)}\n")
        
        for idx, row in enumerate(self.raw_rows):
            print(f"Row {idx+1:3d} (y={row['y_position']:4d}, conf={row['avg_confidence']:.1f}%): {row['full_text']}")
        
        return self.raw_rows

    def create_table_structure(self, start_row=17, header_keywords=None):
        """
        Create a structured table by organizing rows into columns starting from a specific row
        start_row: Row number to start table extraction (1-indexed, default=17)
        header_keywords: List of keywords to identify header row (e.g., ['NAMES', 'ADDRESS', 'DATE'])
        """
        if header_keywords is None:
            header_keywords = ['NAME', 'NAMES', 'ADDRESS', 'DATE', 'AMOUNT', 'CHILDREN', 'GUARDIAN']
        
        # Use scaled boundaries for word coordinate comparison
        filtered_boundaries = [x * self.scale_factor for x in self.column_boundaries if x not in self.EXCLUDED_COLUMNS]
        filtered_boundaries = [int(x) for x in filtered_boundaries]  # Convert to int
        
        if not filtered_boundaries:
            print("[WARN] No column boundaries detected — creating single column table")
            # If no columns detected, create simple table with full text
            table_data = []
            for idx, row in enumerate(self.raw_rows):
                if idx >= start_row - 1:  # Convert to 0-indexed
                    table_data.append([row['full_text']])
            return pd.DataFrame(table_data, columns=['Text'])
        
        print(f"\n[INFO] Original column boundaries (filtered): {[x for x in self.column_boundaries if x not in self.EXCLUDED_COLUMNS]}")
        print(f"[INFO] Scaled column boundaries (for word matching): {filtered_boundaries}")
        print(f"[INFO] Excluded boundaries: {self.EXCLUDED_COLUMNS}")
        print(f"[INFO] Starting table extraction from row {start_row}")
        
        # Use predefined column names
        num_expected_columns = len(filtered_boundaries) + 1
        
        # If we have predefined column names and they match the expected number of columns
        if len(self.column_names) == num_expected_columns:
            headers = self.column_names
            print(f"[INFO] Using predefined column names: {headers}")
        else:
            # Try to find header row
            header_row_idx = max(0, start_row - 4)
            search_start = max(0, start_row - 4)
            search_end = min(len(self.raw_rows), start_row + 2)
            
            for idx in range(search_start, search_end):
                if idx < len(self.raw_rows):
                    row_text = self.raw_rows[idx]['full_text'].upper()
                    if any(keyword in row_text for keyword in header_keywords):
                        header_row_idx = idx
                        print(f"[INFO] Header row detected at row {idx + 1}: {self.raw_rows[idx]['full_text']}")
                        break
            
            # Create column headers from header row
            if header_row_idx < len(self.raw_rows):
                header_row = self.raw_rows[header_row_idx]
                headers = self._distribute_words_to_columns(header_row['words'], filtered_boundaries)
                headers = [h.strip() or f"Column_{i+1}" for i, h in enumerate(headers)]
            else:
                # Use default column names
                headers = [f"Column_{i+1}" for i in range(num_expected_columns)]
            
            print(f"[INFO] Detected column names: {headers}")
        
        print(f"[INFO] Number of columns: {len(headers)}")
        
        # Process data rows starting from specified row
        table_data = []
        data_start_row = start_row - 1  # Convert to 0-indexed
        
        print(f"[INFO] Extracting data from row {data_start_row + 1} to row {len(self.raw_rows)}")
        
        for idx in range(data_start_row, len(self.raw_rows)):
            row = self.raw_rows[idx]
            row_cells = self._distribute_words_to_columns(row['words'], filtered_boundaries)
            table_data.append(row_cells)
            
            # Debug: Show first few rows assignment
            if idx - data_start_row < 5:
                print(f"  Row {idx + 1}: {row_cells}")
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=headers)
        
        print(f"\n{'='*80}")
        print(f"TABLE STRUCTURE")
        print(f"{'='*80}")
        print(f"Start row: {start_row}")
        print(f"Columns: {headers}")
        print(f"Data rows: {len(table_data)}")
        print(f"\nColumn boundaries used for alignment (scaled coordinates):")
        for i, boundary in enumerate(filtered_boundaries):
            col_range_start = 0 if i == 0 else filtered_boundaries[i-1]
            original_boundary = int(boundary / self.scale_factor)
            print(f"  Column '{headers[i]}': x < {boundary} (original: x < {original_boundary})")
        original_last = int(filtered_boundaries[-1] / self.scale_factor)
        print(f"  Column '{headers[-1]}': x >= {filtered_boundaries[-1]} (original: x >= {original_last})")
        print(f"\nPreview (first 10 rows):")
        print(df.head(10).to_string(index=False))
        print(f"{'='*80}\n")
        
        return df

    def _distribute_words_to_columns(self, words, boundaries):
        """
        Distribute words into columns based on their x-position relative to vertical boundaries.
        Words with x < boundary go to the left column, words with x >= boundary go to the right.
        """
        num_columns = len(boundaries) + 1
        columns = [""] * num_columns
        
        # Sort words by x position for consistent ordering within each column
        sorted_words = sorted(words, key=lambda w: w[0])
        
        for x, word, conf in sorted_words:
            col_idx = self._get_column_index(x, boundaries)
            if col_idx < num_columns:
                # Add space between words in the same column
                if columns[col_idx]:
                    columns[col_idx] += " "
                columns[col_idx] += word
        
        return [col.strip() for col in columns]

    def _get_column_index(self, x, boundaries):
        """
        Determine which column an x-coordinate belongs to based on vertical boundaries.
        Text with x < boundary[i] belongs to column i.
        Text with x >= boundary[-1] belongs to the last column.
        """
        for i, boundary in enumerate(boundaries):
            if x < boundary:
                return i
        # If x is greater than all boundaries, it belongs to the rightmost column
        return len(boundaries)

    def isolate_vertical_lines(self, binarized=None):
        if binarized is None:
            binarized = self.gray
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, binarized.shape[0] // 20)))
        temp_img = cv2.erode(binarized, vertical_kernel, iterations=1)
        vertical_lines_img = cv2.dilate(temp_img, vertical_kernel, iterations=2)
        return vertical_lines_img

    def draw_vertical_lines(self, color=(255, 0, 0), thickness=2, draw_isolated=True, min_line_height=1000):
        """Draw vertical lines on the original image (not scaled)"""
        filtered_boundaries = [x for x in self.column_boundaries if x not in self.EXCLUDED_COLUMNS]
        if filtered_boundaries:
            h = self.img.shape[0]
            for x in filtered_boundaries:
                cv2.line(self.img, (x, 0), (x, h), color, thickness)

        if draw_isolated:
            # Create a temporary grayscale version of original image for line detection
            gray_original = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            isolated_lines_img = self.isolate_vertical_lines(gray_original)
            contours, _ = cv2.findContours(isolated_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h >= min_line_height and w < 8:
                    cv2.line(self.img, (x + w // 2, y), (x + w // 2, y + h), (0, 0, 255), 1)

    def reconstruct_text(self):
        """Reconstruct text from line groups"""
        sorted_lines = sorted(self.line_groups.items(), key=lambda kv: kv[0])
        final_text_lines = []
        
        for _, words in sorted_lines:
            words = sorted(words, key=lambda w: w[0])
            line_text = " ".join([w[1] for w in words])
            final_text_lines.append(line_text)
        
        self.final_text = "\n".join(final_text_lines)
        return self.final_text

    def save_results(self, output_dir="outputs"):
        """Save text and image results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw text
        text_path = os.path.join(output_dir, "output_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(self.final_text)
        print(f"[INFO] Text saved at: {text_path}")
        
        # Save row-wise text
        rows_path = os.path.join(output_dir, "output_rows.txt")
        with open(rows_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"ROW-WISE TEXT EXTRACTION\n")
            f.write(f"{'='*80}\n\n")
            for idx, row in enumerate(self.raw_rows):
                f.write(f"Row {idx+1:3d} (y={row['y_position']:4d}, conf={row['avg_confidence']:.1f}%): {row['full_text']}\n")
        print(f"[INFO] Row-wise text saved at: {rows_path}")
        
        # Save image with bounding boxes
        bbox_image_path = os.path.join(output_dir, "output_boxes.jpg")
        cv2.imwrite(bbox_image_path, self.img)
        print(f"[INFO] Image with bounding boxes saved at: {bbox_image_path}")

    def save_table_to_excel(self, df, output_path="outputs/table_extracted.xlsx"):
        """Save structured table to Excel"""
        if df is None or df.empty:
            print("[WARN] No table data available to save.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extracted Table', index=False)
        
        print(f"[INFO] Table data saved to: {output_path}")

    def save_table_to_csv(self, df, output_path="outputs/table_extracted.csv"):
        """Save structured table to CSV"""
        if df is None or df.empty:
            print("[WARN] No table data available to save.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"[INFO] Table data saved to: {output_path}")

    def save_table_to_text(self, df, output_path="outputs/table_structure.txt"):
        """Save structured table to a formatted text file"""
        if df is None or df.empty:
            print("[WARN] No table data available to save.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("="*100 + "\n")
            f.write("TABLE STRUCTURE - EXTRACTED DATA\n")
            f.write("="*100 + "\n\n")
            
            # Write column information
            f.write(f"Number of columns: {len(df.columns)}\n")
            f.write(f"Number of rows: {len(df)}\n")
            f.write(f"Columns: {', '.join(df.columns)}\n")
            f.write("\n" + "="*100 + "\n\n")
            
            # Calculate column widths for formatting
            col_widths = {}
            for col in df.columns:
                max_width = max(
                    len(str(col)),
                    df[col].astype(str).str.len().max() if not df[col].empty else 0
                )
                col_widths[col] = min(max_width + 2, 30)  # Max width 30 chars
            
            # Write header row
            header_line = " | ".join([str(col).ljust(col_widths[col]) for col in df.columns])
            f.write(header_line + "\n")
            f.write("-"*len(header_line) + "\n")
            
            # Write data rows
            for idx, row in df.iterrows():
                row_line = " | ".join([
                    str(row[col])[:col_widths[col]].ljust(col_widths[col]) 
                    for col in df.columns
                ])
                f.write(row_line + "\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write("="*100 + "\n")
        
        print(f"[INFO] Table structure saved to: {output_path}")

    def refine_table_data(self, df):
        """
        Refine and clean the extracted table data based on column types
        - Name columns: Only letters and spaces
        - Age column: Only numbers
        - Address column: Remove special characters like *, |, ", :
        - Guardian column: Only letters and spaces
        """
        if df is None or df.empty:
            print("[WARN] No table data to refine.")
            return df
        
        print(f"\n{'='*80}")
        print(f"REFINING TABLE DATA")
        print(f"{'='*80}\n")
        
        refined_df = df.copy()
        
        # Define column type mappings
        column_rules = {
            'No': 'number',
            'Name of children': 'letters_only',
            'Age': 'number',
            'Address': 'address',
            'Name of guardian or parent': 'letters_only'
        }
        
        for col in refined_df.columns:
            if col in column_rules:
                rule_type = column_rules[col]
                print(f"[INFO] Cleaning column '{col}' using rule: {rule_type}")
                
                if rule_type == 'number':
                    refined_df[col] = refined_df[col].apply(self._extract_numbers)
                elif rule_type == 'letters_only':
                    refined_df[col] = refined_df[col].apply(self._extract_letters_only)
                elif rule_type == 'address':
                    refined_df[col] = refined_df[col].apply(self._clean_address)
                
                # Show sample of changes
                sample_idx = min(3, len(refined_df))
                if sample_idx > 0:
                    print(f"  Sample transformations for '{col}':")
                    for i in range(sample_idx):
                        original = str(df.iloc[i][col])
                        cleaned = str(refined_df.iloc[i][col])
                        if original != cleaned:
                            print(f"    Row {i+1}: '{original}' → '{cleaned}'")
        
        print(f"\n{'='*80}")
        print(f"DATA REFINEMENT COMPLETE")
        print(f"{'='*80}\n")
        
        return refined_df
    
    def _extract_numbers(self, text):
        """Extract only numbers from text"""
        import re
        if pd.isna(text) or text == '':
            return ''
        # Extract all digits
        numbers = re.findall(r'\d+', str(text))
        return ''.join(numbers)
    
    def _extract_letters_only(self, text):
        """Extract only letters and spaces from text"""
        import re
        if pd.isna(text) or text == '':
            return ''
        # Keep only letters and spaces, remove numbers and special characters
        cleaned = re.sub(r'[^a-zA-Z\s]', '', str(text))
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def _clean_address(self, text):
        """Clean address by removing unwanted special characters"""
        import re
        if pd.isna(text) or text == '':
            return ''
        # Remove specific unwanted characters: *, |, ", :, and other problematic ones
        text = str(text)
        # Remove: * | " : ~ ` ^ { } [ ] < >
        unwanted_chars = ['*', '|', '"', ':', '~', '`', '^', '{', '}', '[', ']', '<', '>', ')', '(']
        for char in unwanted_chars:
            text = text.replace(char, '')
        
        # Remove excessive special characters but keep basic punctuation (. , - /)
        text = re.sub(r'[^\w\s.,\-/()&]', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()


# -------------------------------------------
# Orchestrator
# -------------------------------------------

class OCR:
    def __init__(self, image_path, output_dir="outputs", preprocessing='advanced', 
                 tesseract_config='--psm 6 --oem 1', min_conf=0, show=False,
                 header_keywords=None, table_start_row=17, refine_data=True):
        self.image_path = image_path
        self.output_dir = output_dir
        self.preprocessing = preprocessing
        self.tesseract_config = tesseract_config
        self.min_conf = min_conf
        self.show = show
        self.header_keywords = header_keywords or ['NAME', 'NAMES', 'ADDRESS', 'DATE', 'AMOUNT']
        self.table_start_row = table_start_row
        self.refine_data = refine_data  # Whether to clean/refine extracted data

    def run(self):
        print(f"\n{'='*80}")
        print(f"OCR PIPELINE STARTED")
        print(f"{'='*80}\n")
        
        # Step 1: Preprocess image
        print("[STEP 1] Preprocessing image...")
        orig_bgr, processed, scale_factor = preprocess_image(
            self.image_path, 
            method=self.preprocessing, 
            save_steps=(self.preprocessing in ['contrast', 'advanced'])
        )
        print(f"✓ Preprocessing complete using method: {self.preprocessing}")
        print(f"✓ Scale factor: {scale_factor}x")
        print(f"✓ Original image size: {orig_bgr.shape[:2]}")
        print(f"✓ Processed image size: {processed.shape[:2]}\n")
        
        # Step 2: Detect vertical lines (column boundaries) on ORIGINAL image
        print("[STEP 2] Detecting column boundaries on original image...")
        column_boundaries = detect_vertical_lines(image_path=self.image_path, binarized=None)
        print(f"✓ Detected {len(column_boundaries)} column boundaries (original coordinates)\n")

        # Step 3: OCR text extraction on PROCESSED image
        print("[STEP 3] Extracting text with OCR from processed image...")
        extractor = OCRExtraction(processed, orig_bgr.copy(), column_boundaries, scale_factor)
        extractor.extract_text(config=self.tesseract_config, min_conf=self.min_conf)
        extractor.draw_vertical_lines()
        print(f"✓ Text extraction complete (coordinates in scaled space)\n")

        # Step 4: Extract rows
        print("[STEP 4] Extracting text row-wise...")
        extractor.extract_rows()
        print(f"✓ Row extraction complete\n")
        
        # Step 5: Reconstruct plain text
        print("[STEP 5] Reconstructing plain text...")
        extractor.reconstruct_text()
        print(f"✓ Text reconstruction complete\n")
        
        # Step 6: Create table structure with coordinate alignment
        print("[STEP 6] Creating table structure with coordinate alignment...")
        df = extractor.create_table_structure(
            start_row=self.table_start_row,
            header_keywords=self.header_keywords
        )
        print(f"✓ Table structure created with proper coordinate mapping\n")
        
        # Step 6.5: Refine data (clean up columns)
        if self.refine_data:
            print("[STEP 6.5] Refining and cleaning table data...")
            df_refined = extractor.refine_table_data(df)
            print(f"✓ Data refinement complete\n")
        else:
            df_refined = df
        
        # Step 7: Save all results
        print("[STEP 7] Saving results...")
        extractor.save_results(self.output_dir)
        
        # Save both raw and refined versions
        extractor.save_table_to_excel(df, f"{self.output_dir}/table_extracted_raw.xlsx")
        extractor.save_table_to_csv(df, f"{self.output_dir}/table_extracted_raw.csv")
        extractor.save_table_to_text(df, f"{self.output_dir}/table_structure_raw.txt")
        
        if self.refine_data:
            extractor.save_table_to_excel(df_refined, f"{self.output_dir}/table_extracted_refined.xlsx")
            extractor.save_table_to_csv(df_refined, f"{self.output_dir}/table_extracted_refined.csv")
            extractor.save_table_to_text(df_refined, f"{self.output_dir}/table_structure_refined.txt")
            print(f"✓ Both raw and refined versions saved to {self.output_dir}\n")
        else:
            print(f"✓ Raw results saved to {self.output_dir}\n")

        # Step 8: Optional display
        if self.show:
            cv2.imshow("Processed Image", processed)
            cv2.imshow("Detected Text Boxes", extractor.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"{'='*80}")
        print(f"OCR PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        
        # Return refined version if available, otherwise raw
        return (df_refined if self.refine_data else df), extractor


# -------------------------------------------
# Quick benchmark
# -------------------------------------------

def quick_benchmark(image_path):
    print("=" * 80)
    print("OCR EXTRACTION WITH MULTIPLE PREPROCESSING OPTIONS")
    print("=" * 80)

    print("\nSTEP 1: Testing preprocessing methods...")
    print("-" * 80)

    preprocessing_methods = [
        ('original', 'No processing'),
        ('grayscale', 'Grayscale conversion'),
        ('contrast', 'Contrast enhancement (CLAHE)'),
        ('bilateral', 'ImagePreprocessor class (bilateral + adaptive)'),
        ('advanced', 'Advanced (upscale + denoise + CLAHE + Otsu)'),
        ('otsu', 'Otsu thresholding'),
        ('adaptive', 'Adaptive thresholding'),
    ]

    print("\nGenerating preprocessed images for comparison...")
    for method, description in preprocessing_methods:
        print(f"  Processing: {description}...")
        try:
            _, processed, scale = preprocess_image(image_path, method=method, save_steps=(method in ['contrast', 'advanced']))
            cv2.imwrite(f'preprocessed_{method}.jpg', processed)
            print(f"    ✓ Saved: preprocessed_{method}.jpg (scale: {scale}x)")
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:50]}")

    print("\n" + "=" * 80)
    print("STEP 2: Testing OCR with different preprocessing methods...")
    print("=" * 80)

    psm_configs = [
        ('--psm 4 --oem 1', 'PSM 4: Single column + LSTM'),
        ('--psm 6 --oem 1', 'PSM 6: Uniform block + LSTM'),
        ('--psm 3 --oem 1', 'PSM 3: Fully automatic + LSTM'),
    ]

    test_preprocessing = ['original', 'grayscale', 'contrast', 'bilateral', 'advanced']

    print("\nTesting combinations of PSM modes and preprocessing...")
    print("-" * 80)

    results = []
    for config, config_desc in psm_configs:
        print(f"\n{config_desc}:")
        for preproc in test_preprocessing:
            try:
                _, processed, scale = preprocess_image(image_path, method=preproc)
                data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=config)
                words = [t for t in data['text'] if (t or '').strip()]
                print(f"  {preproc:20s}: {len(words):4d} words detected (scale: {scale}x)")
                results.append({'config': config, 'preproc': preproc, 'count': len(words), 'description': config_desc})
            except Exception as e:
                print(f"  {preproc:20s}: Error - {str(e)[:50]}")

    if results:
        best = max(results, key=lambda x: x['count'])
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION FOUND:")
        print("=" * 80)
        print(f"  Tesseract config: {best['config']}")
        print(f"  Preprocessing: {best['preproc']}")
        print(f"  Words detected: {best['count']}")
        print("=" * 80)


if __name__ == "__main__":
    # Example usage
    img_path = "img.jpg"

    # Option 1: Run full pipeline with data refinement
    pipeline = OCR(
        image_path=img_path,
        output_dir="outputs",
        preprocessing='advanced',
        tesseract_config='--psm 6 --oem 1',
        min_conf=0,
        show=False,
        header_keywords=['NAME', 'CHILDREN', 'AGE', 'ADDRESS', 'GUARDIAN', 'PARENT'],
        table_start_row=17,
        refine_data=True  # Enable data refinement/cleaning
    )
    df, extractor = pipeline.run()
    
    # Access the results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows (refined):")
    print(df.head())
    print("="*80)
    
    print("\nOutput files created:")
    print("  - table_extracted_raw.xlsx (original OCR output)")
    print("  - table_extracted_refined.xlsx (cleaned data)")
    print("  - table_structure_raw.txt (original text table)")
    print("  - table_structure_refined.txt (cleaned text table)")
    
    # Option 2: Run benchmark (uncomment to use)
    # quick_benchmark(img_path)