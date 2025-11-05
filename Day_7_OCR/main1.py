import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
import os

class ImagePreprocessor:
    """
    Handles all image preprocessing tasks:
    Reading image, Converting to grayscale, Removing noises,
    Thresholding, Resizing
    """
 
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
 
    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            print(f"Error: Could not read image file '{self.image_path}'.")
        return self.original_image
 
    def to_grayscale(self):
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return self.processed_image
   
    def remove_noise(self):
        # Bilateral filter - preserves edges while removing noise
        self.processed_image = cv2.bilateralFilter(self.processed_image, 4, 1, 25)
        return self.processed_image
 
    def apply_threshold(self):
        # Adaptive thresholding
        self.processed_image = cv2.adaptiveThreshold(
            self.processed_image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 18
        )
        return self.processed_image
 
    def resize_images(self, scale=1.5):
        self.processed_image = cv2.resize(
            self.processed_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        self.original_image = cv2.resize(
            self.original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        return self.processed_image, self.original_image
   
    def save_processed_image(self, output_dir="out"):
        """Save the processed grayscale image."""
        os.makedirs(output_dir, exist_ok=True)
        processed_path = os.path.join(output_dir, "processed_image.jpg")
        cv2.imwrite(processed_path, self.processed_image)
        print(f"[INFO] Processed image saved at: {processed_path}")
 
    def preprocess(self):
        self.load_image()
        self.to_grayscale()
        self.remove_noise()
        self.apply_threshold()
        return self.resize_images()

def enhance_contrast(image):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Makes text stand out more clearly from the background
    Returns the enhanced grayscale image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE - works well for documents with varying lighting
    # clipLimit: Threshold for contrast limiting (higher = more contrast)
    # tileGridSize: Size of grid for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def preprocess_image(image_path, method='original', save_steps=False):
    """
    Preprocessing for document OCR
    
    Available methods:
    - 'original': No processing (best for high-quality scans)
    - 'grayscale': Just grayscale conversion
    - 'contrast': Enhance contrast using CLAHE
    - 'bilateral': ImagePreprocessor class method (bilateral filter + adaptive threshold)
    - 'advanced': Upscaling + denoising + CLAHE + Otsu
    - 'otsu': Binary thresholding with Otsu's method
    - 'adaptive': Adaptive thresholding (good for uneven lighting)
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    if method == 'original':
        # NO PROCESSING - Use original image directly
        processed = img
        if save_steps:
            cv2.imwrite('step1_original_kept.jpg', img)
    
    elif method == 'grayscale':
        # Just convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = gray
        if save_steps:
            cv2.imwrite('step1_grayscale.jpg', gray)
    
    elif method == 'contrast':
        # CONTRAST ENHANCEMENT using CLAHE
        if save_steps:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('step1_grayscale.jpg', gray)
        
        enhanced = enhance_contrast(img)
        processed = enhanced
        
        if save_steps:
            cv2.imwrite('step2_contrast_enhanced.jpg', enhanced)
    
    elif method == 'bilateral':
        # USE ImagePreprocessor CLASS METHOD
        preprocessor = ImagePreprocessor(image_path)
        processed, _ = preprocessor.preprocess()
        
        if save_steps:
            cv2.imwrite('step1_bilateral_processed.jpg', processed)
        
        preprocessor.save_processed_image()
    
    elif method == 'advanced':
        # ADVANCED PREPROCESSING: Upscale + Denoise + CLAHE + Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if save_steps:
            cv2.imwrite('step1_grayscale.jpg', gray)
        
        # Resize for better OCR (200% increase)
        scale_percent = 200
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        if save_steps:
            cv2.imwrite('step2_resized.jpg', resized)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(resized, h=10)
        
        if save_steps:
            cv2.imwrite('step3_denoised.jpg', denoised)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        if save_steps:
            cv2.imwrite('step4_enhanced.jpg', enhanced)
        
        # Otsu's thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if save_steps:
            cv2.imwrite('step5_otsu.jpg', thresh)
        
        processed = thresh
        
    elif method == 'otsu':
        # Grayscale + Otsu thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed = thresh
        if save_steps:
            cv2.imwrite('step1_grayscale.jpg', gray)
            cv2.imwrite('step2_otsu.jpg', thresh)
        
    elif method == 'adaptive':
        # Grayscale + Adaptive thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed = thresh
        if save_steps:
            cv2.imwrite('step1_grayscale.jpg', gray)
            cv2.imwrite('step2_adaptive.jpg', thresh)
    
    else:
        # Default: keep original
        processed = img
    
    return img, processed

def upscale_image(image_path, scale_factor=2):
    """
    Upscale image for better OCR results
    Scale factor of 2-3 usually works well
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    height, width = img.shape[:2]
    new_dimensions = (width * scale_factor, height * scale_factor)
    
    # Use INTER_CUBIC for upscaling (better quality)
    upscaled = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_CUBIC)
    
    # Save upscaled image
    upscaled_path = image_path.rsplit('.', 1)[0] + '_upscaled.jpg'
    cv2.imwrite(upscaled_path, upscaled)
    
    print(f"Upscaled image saved to: {upscaled_path}")
    print(f"Original size: {width}x{height}, New size: {new_dimensions[0]}x{new_dimensions[1]}")
    
    return upscaled_path

def extract_text_with_boxes(image_path, output_image_path='output_with_boxes.jpg', 
                           config='--psm 6', min_conf=0, preprocessing='enhanced'):
    """
    Extract text using pytesseract and draw bounding boxes
    
    PSM modes:
    0 = Orientation and script detection (OSD) only
    1 = Automatic page segmentation with OSD
    3 = Fully automatic page segmentation, but no OSD (Default)
    4 = Assume a single column of text of variable sizes
    5 = Assume a single uniform block of vertically aligned text
    6 = Assume a single uniform block of text
    7 = Treat the image as a single text line
    8 = Treat the image as a single word
    9 = Treat the image as a single word in a circle
    10 = Treat the image as a single character
    11 = Sparse text. Find as much text as possible in no particular order
    12 = Sparse text with OSD
    13 = Raw line. Treat the image as a single text line
    """
    # Use specified preprocessing method
    original_img = cv2.imread(image_path)
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Get bounding boxes with custom config
    data = pytesseract.image_to_data(processed_img, config=config, 
                                     output_type=pytesseract.Output.DICT)
    
    # Draw bounding boxes on original image
    img_with_boxes = original_img.copy()
    n_boxes = len(data['text'])
    
    detected_words = []
    
    for i in range(n_boxes):
        # Filter by confidence
        if int(data['conf'][i]) > min_conf:
            text = data['text'][i].strip()
            if text:
                (x, y, w, h) = (data['left'][i], data['top'][i], 
                               data['width'][i], data['height'][i])
                
                detected_words.append({
                    'text': text,
                    'conf': data['conf'][i],
                    'bbox': (x, y, w, h)
                })
                
                # Draw rectangle
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Put text above the box (smaller font for dense documents)
                cv2.putText(img_with_boxes, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, img_with_boxes)
    print(f"Image with bounding boxes saved to: {output_image_path}")
    print(f"Detected {len(detected_words)} words/tokens")
    
    return data, img_with_boxes, detected_words

def extract_text_line_boxes(image_path, output_image_path='output_with_line_boxes.jpg'):
    """
    Extract text using pytesseract and draw bounding boxes for lines
    """
    original_img, processed_img = preprocess_image(image_path)
    
    # Method 2: Get bounding boxes for each line
    data = pytesseract.image_to_boxes(processed_img)
    
    img_with_boxes = original_img.copy()
    h, w, _ = img_with_boxes.shape
    
    for box in data.splitlines():
        box = box.split()
        if len(box) == 6:
            char, x, y, x2, y2, conf = box
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
            
            # Convert coordinates (tesseract uses bottom-left origin)
            y = h - y
            y2 = h - y2
            
            cv2.rectangle(img_with_boxes, (x, y2), (x2, y), (0, 0, 255), 1)
            cv2.putText(img_with_boxes, char, (x, y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite(output_image_path, img_with_boxes)
    print(f"Image with character boxes saved to: {output_image_path}")
    
    return data, img_with_boxes

def extract_text_by_lines(image_path, config='--psm 6', min_conf=0, preprocessing='enhanced'):
    """
    Extract text organized by lines, preserving the original layout
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Get detailed data
    data = pytesseract.image_to_data(processed_img, config=config,
                                     output_type=pytesseract.Output.DICT)
    
    # Group words by line
    lines = {}
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        if int(data['conf'][i]) > min_conf:
            text = data['text'][i].strip()
            if text:
                # Use block_num and line_num to group words on same line
                block_num = data['block_num'][i]
                line_num = data['line_num'][i]
                key = (block_num, line_num)
                
                if key not in lines:
                    lines[key] = []
                
                lines[key].append({
                    'text': text,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'conf': data['conf'][i]
                })
    
    # Sort lines by vertical position (top coordinate)
    sorted_lines = []
    for key in sorted(lines.keys()):
        # Sort words in each line by horizontal position (left coordinate)
        line_words = sorted(lines[key], key=lambda x: x['left'])
        line_text = ' '.join([word['text'] for word in line_words])
        avg_top = sum([word['top'] for word in line_words]) / len(line_words)
        sorted_lines.append((avg_top, line_text))
    
    # Sort by vertical position
    sorted_lines.sort(key=lambda x: x[0])
    
    # Return text with each line on a new line
    return '\n'.join([line[1] for line in sorted_lines])

def extract_text_by_lines_detailed(image_path, config='--psm 6', min_conf=0, preprocessing='enhanced'):
    """
    Extract text organized by lines with detailed information
    Returns list of lines with their positions and words
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Get detailed data
    data = pytesseract.image_to_data(processed_img, config=config,
                                     output_type=pytesseract.Output.DICT)
    
    # Group words by line
    lines = {}
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        if int(data['conf'][i]) > min_conf:
            text = data['text'][i].strip()
            if text:
                block_num = data['block_num'][i]
                line_num = data['line_num'][i]
                key = (block_num, line_num)
                
                if key not in lines:
                    lines[key] = {
                        'words': [],
                        'block': block_num,
                        'line': line_num
                    }
                
                lines[key]['words'].append({
                    'text': text,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })
    
    # Sort and format lines
    result_lines = []
    for key in sorted(lines.keys()):
        line_data = lines[key]
        # Sort words in each line by horizontal position
        line_data['words'].sort(key=lambda x: x['left'])
        
        # Calculate line position (average top of all words)
        avg_top = sum([word['top'] for word in line_data['words']]) / len(line_data['words'])
        avg_left = min([word['left'] for word in line_data['words']])
        
        line_text = ' '.join([word['text'] for word in line_data['words']])
        
        result_lines.append({
            'text': line_text,
            'top': avg_top,
            'left': avg_left,
            'block': line_data['block'],
            'line': line_data['line'],
            'words': line_data['words']
        })
    
    # Sort by vertical position
    result_lines.sort(key=lambda x: x['top'])
    
    return result_lines

def extract_text_with_confidence(image_path, preprocessing='original'):
    """
    Extract text with confidence scores for each word
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Convert to PIL Image
    if len(processed_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(processed_img)
    
    custom_config = r'--oem 3 --psm 6'
    
    # Get detailed data including confidence scores
    data = pytesseract.image_to_data(
        pil_img, 
        config=custom_config, 
        output_type=pytesseract.Output.DICT
    )
    
    # Filter out low confidence results
    results = []
    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        text = data['text'][i].strip()
        
        if conf > 0 and text:  # Only include confident results
            results.append({
                'text': text,
                'confidence': conf,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
    
    return results


def detect_table_structure(image_path, preprocessing='original'):
    """
    Detect table structure by analyzing text positions
    Returns column boundaries and row positions
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Convert to PIL Image
    if len(processed_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(processed_img)
    
    custom_config = r'--oem 1 --psm 6'
    
    # Get detailed word positions
    data = pytesseract.image_to_data(
        pil_img, 
        config=custom_config, 
        output_type=pytesseract.Output.DICT
    )
    
    # Collect all x-coordinates (left positions) of words
    x_positions = []
    y_positions = []
    
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            x_positions.append(data['left'][i])
            y_positions.append(data['top'][i])
    
    if not x_positions:
        return [], []
    
    # Cluster x-positions to find column boundaries
    x_positions = sorted(x_positions)
    y_positions = sorted(y_positions)
    
    # Find column breaks (gaps in x-positions)
    column_breaks = [x_positions[0]]
    gap_threshold = 30  # Minimum gap to consider a new column
    
    for i in range(1, len(x_positions)):
        if x_positions[i] - x_positions[i-1] > gap_threshold:
            column_breaks.append(x_positions[i])
    
    # Find row breaks (gaps in y-positions)
    row_breaks = [y_positions[0]]
    row_gap_threshold = 15  # Minimum gap to consider a new row
    
    prev_y = y_positions[0]
    for y in y_positions[1:]:
        if y - prev_y > row_gap_threshold:
            row_breaks.append(y)
            prev_y = y
    
    return column_breaks, row_breaks

def save_table_to_markdown(table_data, file_path="ocr_extracted_table.txt"):
    """
    Save table data in Markdown format with uniform column spacing
    
    Parameters:
    - table_data: List of dictionaries with column headers as keys
    - file_path: Output file path
    """
    if not table_data:
        print("⚠️ No data to save.")
        return
    
    headers = list(table_data[0].keys())
    
    # Calculate maximum width for each column
    col_widths = {}
    for header in headers:
        # Start with header length
        max_width = len(header)
        # Check all rows for maximum content width
        for row in table_data:
            value = str(row.get(header, ''))
            max_width = max(max_width, len(value))
        # Add padding (minimum 3 characters, maximum 50 to avoid extremely wide columns)
        col_widths[header] = min(max(max_width + 2, 5), 50)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write header with uniform spacing
        header_parts = []
        separator_parts = []
        for header in headers:
            width = col_widths[header]
            header_parts.append(header.ljust(width))
            separator_parts.append('-' * width)
        
        f.write('| ' + ' | '.join(header_parts) + ' |\n')
        f.write('| ' + ' | '.join(separator_parts) + ' |\n')
        
        # Write rows with uniform spacing
        for row in table_data:
            row_parts = []
            for header in headers:
                value = str(row.get(header, ''))
                width = col_widths[header]
                # Truncate if too long
                if len(value) > width:
                    value = value[:width-3] + '...'
                row_parts.append(value.ljust(width))
            f.write('| ' + ' | '.join(row_parts) + ' |\n')
    
    print(f"✅ Table successfully saved in Markdown-style format: {file_path}")
    print(f"   Column widths: {', '.join([f'{h}: {w}' for h, w in col_widths.items()])}")


def assign_text_to_columns(detailed_lines, x_positions, census_headers, 
                           start_line=12, start_y_position=389):
    """
    Organize OCR text lines into a table structure using detected vertical divider positions.
    Each line of text is split across columns based on word positions.
    
    Parameters:
    - detailed_lines: List of lines with words and positions (from extract_text_by_lines_detailed)
    - x_positions: List of x-coordinates of vertical lines (column dividers)
    - census_headers: List of column headers
    - start_line: Line number to start from (default: 12, skips first 11 lines)
    - start_y_position: Y-coordinate to start from (default: 389)
    
    Returns:
    - List of dictionaries with column headers as keys
    """
    # Ensure we have exactly 4 dividers for 5 columns
    x_positions = sorted(x_positions[:4])
    print("x coordinates:", x_positions)
    print("??????????????????????????????????"*80)

    # print("detailed lines", detailed_lines)
    print("??????????????????????????????????"*80)

    
    print(f"\n{'='*80}")
    print("ASSIGNING TEXT TO COLUMNS BASED ON VERTICAL LINES")
    print(f"{'='*80}")
    print(f"Filtering criteria:")
    print(f"  - Start from line number: {start_line} (skipping first {start_line-1} lines)")
    print(f"  - Start from Y-position: {start_y_position}")
    print(f"Column dividers at x-positions: {x_positions}")
    
    # Filter lines to exclude first 11 lines (start from line 12)
    filtered_lines = []
    for line in detailed_lines:
        if line['line'] >= start_line and line['top'] >= start_y_position:
            filtered_lines.append(line)

    # print("filetered lines:",filtered_lines)
    
    print(f"\nFiltering results:")
    print(f"  Total lines detected: {len(detailed_lines)}")
    print(f"  Lines filtered out (headers/title): {len(detailed_lines) - len(filtered_lines)}")
    print(f"  Lines to process (table data): {len(filtered_lines)}")
    
    if not filtered_lines:
        print("⚠️  Warning: No lines remaining after filtering!")
        return []
    
    # Compute column boundaries (left, right) for each column
    column_boundaries = []
    column_boundaries.append((0, x_positions[0]))  # Column 0: before 1st line
    for i in range(len(x_positions) - 1):
        column_boundaries.append((x_positions[i], x_positions[i+1]))  # Between lines
    column_boundaries.append((x_positions[-1], float('inf')))  # Last column: after last line
    
    print("\nColumn boundary assignments:")
    for idx, (left, right) in enumerate(column_boundaries):
        left_str = f"{left}" if left != 0 else "0"
        right_str = f"{right}" if right != float('inf') else "∞"
        print(f"  Column {idx+1} ({census_headers[idx]}): x in [{left_str}, {right_str})")
    
    # Prepare table structure
    table = []
    
    print(f"\nProcessing {len(filtered_lines)} lines...")
    
    for line_idx, line in enumerate(filtered_lines):
        row = {header: "" for header in census_headers}
        
        for word in line['words']:
            # print("word=", word)
            word_center = word['left'] 

            # Find which column this word belongs to
            for col_idx, (left_bound, right_bound) in enumerate(column_boundaries):
                # print("left bound:",left_bound)
                # print("right bound:",right_bound)
                # print("word text:", word['text'])
                if left_bound <= word_center < right_bound:
                    header = census_headers[col_idx]
                    if row[header]:
                        row[header] += " "
                    row[header] += word['text']
                    break
        
        # Clean up extra spaces
        for header in row:
            row[header] = row[header].strip()
        
        # Add non-empty rows only
        if any(row.values()):
            table.append(row)
    
    print(f"✓ Created table with {len(table)} rows (excluding first {start_line-1} header lines)")
    
    # Show statistics
    print("\nColumn statistics:")
    for header in census_headers:
        non_empty = sum(1 for row in table if row.get(header, '').strip())
        print(f"  {header}: {non_empty} non-empty entries")
    
    return table

# def assign_text_to_columns(detailed_lines, x_positions, census_headers,
#                            start_line=12, start_y_position=389,
#                            merge_threshold=60,   # px: merge divider x's closer than this
#                            blob_gap=25,          # px: gap to separate word blobs on same line
#                            prefer_direction="right"  # "right" or "left" fallback when col occupied
#                           ):
#     """
#     Improved assignment of OCR text into columns.

#     Parameters:
#     - detailed_lines: list of dicts with keys: 'line' (int), 'top' (y), 'words' (list of {'text','left'}).
#     - x_positions: list of detected vertical divider x-coordinates.
#     - census_headers: list of column headers (length = number of columns).
#     - start_line: minimum line number to include.
#     - start_y_position: minimum top y to include.
#     - merge_threshold: px distance under which divider x's are merged.
#     - blob_gap: px horizontal gap used to split words into blobs (per line).
#     - prefer_direction: direction to try when original column occupied ("right" or "left").

#     Returns:
#     - table: list of rows (dicts mapping header -> text)
#     """


#     # --- 1) sanitize and merge x_positions (so close coordinates form a single divider) ---
#     if not x_positions:
#         raise ValueError("x_positions is empty")

#     x_sorted = sorted(x_positions)
#     merged = []
#     current = x_sorted[0]
#     group = [current]

#     for x in x_sorted[1:]:
#         if x - group[-1] <= merge_threshold:
#             group.append(x)
#         else:
#             # use median of the group as representative
#             rep = sorted(group)[len(group)//2]
#             merged.append(rep)
#             group = [x]
#     if group:
#         rep = sorted(group)[len(group)//2]
#         merged.append(rep)

#     # Make sure we have at most len(census_headers)-1 dividers (allow fewer; we'll still compute columns).
#     max_dividers = max(0, len(census_headers) - 1)
#     merged = merged[:max_dividers]
#     # if fewer dividers detected than expected, that's fine; columns will be computed accordingly.

#     # --- 2) compute column centers and boundaries (Voronoi by divider midpoints) ---
#     # column centers: if there are N dividers, there are N+1 column centers:
#     # - centers: left-of-first, between each pair, right-of-last. For centers we can pick midpoints between dividers
#     #   and also extrapolate for edges.
#     divs = merged
#     centers = []

#     if not divs:
#         # single column fallback: whole page
#         centers = [0.0]  # all text goes to column 0
#         boundaries = [(-float("inf"), float("inf"))]
#     else:
#         # compute interior midpoints to define boundaries later; but first compute rough centers
#         # We'll use:
#         #   center0 = divs[0] / 2  (mid between 0 and first divider)
#         #   center_i = (divs[i-1] + divs[i]) / 2  for i in 1..len(divs)-1
#         #   center_last = divs[-1] + (avg_gap or 50)
#         avg_gap = None
#         if len(divs) >= 2:
#             gaps = [divs[i+1] - divs[i] for i in range(len(divs)-1)]
#             avg_gap = sum(gaps) / len(gaps)
#         else:
#             avg_gap = 100  # heuristic for right-most extension if only one divider

#         centers.append(divs[0] / 2.0)
#         for i in range(1, len(divs)):
#             centers.append((divs[i-1] + divs[i]) / 2.0)
#         centers.append(divs[-1] + avg_gap / 2.0)

#         # compute boundaries as midpoints between centers
#         boundaries = []
#         for i in range(len(centers)):
#             left = -float("inf") if i == 0 else (centers[i-1] + centers[i]) / 2.0
#             right = float("inf") if i == len(centers)-1 else (centers[i] + centers[i+1]) / 2.0
#             boundaries.append((left, right))

#     # If number of centers doesn't match number of headers, try aligning:
#     # Case: more headers than centers -> extend centers by spacing; fewer headers -> trim to header count.
#     if len(centers) != len(census_headers):
#         # If centers less than headers, spread remaining evenly to the right
#         if len(centers) < len(census_headers):
#             needed = len(census_headers) - len(centers)
#             last_center = centers[-1]
#             step = avg_gap if avg_gap and avg_gap > 0 else 100
#             for i in range(1, needed+1):
#                 centers.append(last_center + step * i)
#             # recompute boundaries
#             boundaries = []
#             for i in range(len(centers)):
#                 left = -float("inf") if i == 0 else (centers[i-1] + centers[i]) / 2.0
#                 right = float("inf") if i == len(centers)-1 else (centers[i] + centers[i+1]) / 2.0
#                 boundaries.append((left, right))
#         else:
#             # more centers than headers: trim to header count (prefer leftmost)
#             centers = centers[:len(census_headers)]
#             boundaries = boundaries[:len(census_headers)]

#     # --- 3) filter lines by start_line and start_y_position ---
#     filtered_lines = [
#         ln for ln in detailed_lines
#         if ln.get('line', 0) >= start_line and ln.get('top', 0) >= start_y_position
#     ]

#     table = []
#     for line in filtered_lines:
#         # initialize empty row
#         row = {h: "" for h in census_headers}

#         # gather words with their x positions (left)
#         words = [(w['left'], w['text']) for w in line.get('words', []) if 'left' in w and 'text' in w]
#         if not words:
#             # nothing to add; skip
#             continue

#         words.sort(key=lambda t: t[0])  # sort by left

#         # --- 3.a cluster words into horizontal blobs using blob_gap ---
#         blobs = []  # each blob: {'centroid_x': float, 'text': str, 'word_positions': [x,...]}
#         current_blob = {'xs': [], 'texts': []}
#         prev_x = None
#         for x, txt in words:
#             if prev_x is None:
#                 current_blob['xs'].append(x); current_blob['texts'].append(txt)
#             else:
#                 if x - prev_x <= blob_gap:
#                     current_blob['xs'].append(x); current_blob['texts'].append(txt)
#                 else:
#                     # finalize previous
#                     xs = current_blob['xs']
#                     centroid = sum(xs) / len(xs)
#                     blobs.append({'centroid_x': centroid, 'text': " ".join(current_blob['texts']), 'xs': xs})
#                     current_blob = {'xs': [x], 'texts': [txt]}
#             prev_x = x
#         # finalize last
#         if current_blob['xs']:
#             xs = current_blob['xs']
#             centroid = sum(xs) / len(xs)
#             blobs.append({'centroid_x': centroid, 'text': " ".join(current_blob['texts']), 'xs': xs})

#         # --- 3.b assign each blob to a column with fallback if occupied ---
#         for blob in blobs:
#             xcent = blob['centroid_x']

#             # find initial column index by boundary containment; fallback to nearest center if abnormal
#             col_idx = None
#             for i, (left_b, right_b) in enumerate(boundaries):
#                 if left_b <= xcent < right_b:
#                     col_idx = i
#                     break
#             if col_idx is None:
#                 # should not happen, but assign to nearest center
#                 distances = [abs(xcent - c) for c in centers]
#                 col_idx = distances.index(min(distances))

#             # if the chosen header already has text in this row, try to find next available column
#             if row[census_headers[col_idx]].strip():
#                 # attempt to move in preferred direction
#                 moved = False
#                 if prefer_direction == "right":
#                     for j in range(col_idx+1, len(census_headers)):
#                         if not row[census_headers[j]].strip():
#                             col_idx = j
#                             moved = True
#                             break
#                     if not moved:
#                         # attempt left as last resort
#                         for j in range(col_idx-1, -1, -1):
#                             if not row[census_headers[j]].strip():
#                                 col_idx = j
#                                 moved = True
#                                 break
#                 else:  # prefer left
#                     for j in range(col_idx-1, -1, -1):
#                         if not row[census_headers[j]].strip():
#                             col_idx = j
#                             moved = True
#                             break
#                     if not moved:
#                         # attempt right as last resort
#                         for j in range(col_idx+1, len(census_headers)):
#                             if not row[census_headers[j]].strip():
#                                 col_idx = j
#                                 moved = True
#                                 break
#                 # if still not moved, we'll append into the original column (concatenate)
#                 # this avoids losing data.

#             header = census_headers[col_idx]
#             if row[header]:
#                 row[header] += " "
#             row[header] += blob['text']

#         # strip spaces and add non-empty rows
#         for h in row:
#             row[h] = row[h].strip()

#         if any(row.values()):
#             table.append(row)

#     return table



def extract_table_with_columns(image_path, preprocessing='original', 
                                manual_headers=None, save_csv=True,
                                start_line=None, start_y_position=None):
    """
    Extract table data organized by columns
    
    Parameters:
    - image_path: Path to image
    - preprocessing: Preprocessing method to use
    - manual_headers: List of column headers if OCR misreads them
                     e.g., ['No.', 'Name of Children', 'Age', 'Address', 'Name of Parent']
    - save_csv: Whether to save as CSV file
    - start_line: Line number to start table extraction from (e.g., 12)
    - start_y_position: Y-coordinate to start from (alternative to start_line)
    
    Returns:
    - Dictionary with column headers as keys and data as values
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Convert to PIL Image
    if len(processed_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(processed_img)
    
    custom_config = r'--oem 1 --psm 6'
    
    # Get detailed data
    data = pytesseract.image_to_data(
        pil_img, 
        config=custom_config, 
        output_type=pytesseract.Output.DICT
    )
    
    # Collect all words with their positions
    words = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            words.append({
                'text': data['text'][i].strip(),
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': data['conf'][i],
                'block_num': data['block_num'][i],
                'line_num': data['line_num'][i]
            })
    
    if not words:
        return {}
    
    # Sort words by position (top to bottom, left to right)
    words.sort(key=lambda w: (w['top'], w['left']))
    
    # Filter words based on start_line or start_y_position
    if start_line is not None:
        print(f"Filtering to include only line {start_line} onwards...")
        words = [w for w in words if w['line_num'] >= start_line]
        print(f"Remaining words after filtering: {len(words)}")
    
    if start_y_position is not None:
        print(f"Filtering to include only Y-position >= {start_y_position}...")
        words = [w for w in words if w['top'] >= start_y_position]
        print(f"Remaining words after filtering: {len(words)}")
    
    if not words:
        print("Warning: No words found after filtering!")
        return {}
    
    # Detect column boundaries
    column_breaks, row_breaks = detect_table_structure(image_path, preprocessing)
    
    if len(column_breaks) < 2:
        print("Warning: Could not detect multiple columns. Using single column.")
        column_breaks = [0, processed_img.shape[1]]
    
    # Create column ranges
    columns = []
    for i in range(len(column_breaks) - 1):
        columns.append({
            'x_start': column_breaks[i],
            'x_end': column_breaks[i + 1],
            'data': []
        })
    # Add last column
    columns.append({
        'x_start': column_breaks[-1],
        'x_end': processed_img.shape[1],
        'data': []
    })
    
    print(f"Detected {len(columns)} columns")
    print(f"Column boundaries (x-coordinates): {column_breaks}")
    
    # Assign words to columns based on x-position
    for word in words:
        word_center = word['left'] + word['width'] / 2
        
        for col in columns:
            if col['x_start'] <= word_center < col['x_end']:
                col['data'].append(word)
                break
    
    # Debug: Show words per column
    print("\nWords per column:")
    for i, col in enumerate(columns):
        print(f"  Column {i+1}: {len(col['data'])} words")
    
    # Organize data by rows within each column
    organized_data = {}
    
    # Use manual headers if provided
    if manual_headers:
        headers = manual_headers
        print(f"\nUsing manual headers: {headers}")
    else:
        # Generate default headers
        headers = [f'Column_{i+1}' for i in range(len(columns))]
        print(f"\nUsing default headers: {headers}")
    
    # Ensure we have enough headers
    while len(headers) < len(columns):
        headers.append(f'Column_{len(headers)+1}')
    
    # Initialize columns in organized_data
    for header in headers:
        organized_data[header] = []
    
    # Group words into rows and assign to columns
    for i, col in enumerate(columns):
        if i >= len(headers):
            break
            
        header = headers[i]
        col_words = col['data']
        
        # Group by rows
        if col_words:
            current_row = []
            current_y = col_words[0]['top']
            
            for word in col_words:
                # If word is on same row (within threshold)
                if abs(word['top'] - current_y) < 20:
                    current_row.append(word['text'])
                else:
                    # New row
                    if current_row:
                        organized_data[header].append(' '.join(current_row))
                    current_row = [word['text']]
                    current_y = word['top']
            
            # Add last row
            if current_row:
                organized_data[header].append(' '.join(current_row))
    
    # Equalize column lengths (pad shorter columns)
    max_length = max(len(col_data) for col_data in organized_data.values()) if organized_data else 0
    for header in organized_data:
        while len(organized_data[header]) < max_length:
            organized_data[header].append('')
    
    print(f"\nExtracted {max_length} rows of data")
    
    # Save as CSV if requested
    if save_csv:
        df = pd.DataFrame(organized_data)
        df.to_csv('table_columns.csv', index=False)
        print(f"✓ Table data saved to 'table_columns.csv'")
    
    return organized_data

def detect_vertical_lines(image_path, min_length=100, min_display_length=500, x_merge_tolerance=15, draw=True):
    """
    Detects vertical lines in an image using Hough Transform.
    Returns sorted list of x-coordinates representing column boundaries.
    """

    image_path = "img.jpg"
    output_path = "detected_lines.jpg"

    # --- Load image and preprocess ---
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # --- Hough Line Transform ---
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    vertical_lines = []
    min_length = 100  # initial minimum for vertical filtering
    x_merge_tolerance = 15  # tolerance for merging nearby lines

    # --- Filter vertical lines ---
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # keep near-vertical and reasonably long lines
            if abs(angle) > 85 and length >= min_length:
                vertical_lines.append((x1, y1, x2, y2, length))

    # --- Sort lines by x position ---
    vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

    # --- Merge nearby vertical lines ---
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

    # --- Draw only lines with length > 500 ---
    display_lines = [line for line in merged_lines if line[4] > 500]

    for x1, y1, x2, y2, length in display_lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(img, f"{int(length)}px", (x1 + 5, min(y1, y2) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Save output ---
    cv2.imwrite(output_path, img)
    print(f"✅ Output saved to: {output_path}")
 
    # --- Extract and return x-coordinates ---
    x_coords = sorted([int((x1 + x2) / 2) for x1, y1, x2, y2, _ in display_lines])
    print("\nFiltered vertical line x-coordinates:", x_coords)

    return x_coords 

def extract_text_by_columns_with_headers(image_path, detailed_lines, x_positions, census_headers, 
                                         output_txt="structured_table.txt", 
                                         output_csv="structured_table.csv", 
                                         preprocessing='original',
                                         start_line=None, start_y_position=None):
    """
    Assigns OCR-detected words to columns based on vertical line positions and column headers
    
    Column assignment:
    - Text before 1st vertical line → 1st column (No.)
    - Text between 1st and 2nd line → 2nd column (Name of Children)
    - Text between 2nd and 3rd line → 3rd column (Age)
    - Text between 3rd and 4th line → 4th column (Address)
    - Text after 4th line → 5th column (Name of Parent/Guardian)
    
    Parameters:
    - image_path: Path to the input image
    - x_positions: X-coordinates of vertical lines (column dividers)
    - census_headers: List of column headers (e.g., ['No.', 'Name of Children', ...])
    - output_txt: Output text file path
    - output_csv: Output CSV file path
    - preprocessing: Preprocessing method to use
    - start_line: Optional line number to start extraction from
    - start_y_position: Optional Y-coordinate to start extraction from
    
    Returns:
    - DataFrame with extracted table data
    """
    # Preprocess image
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Convert to PIL Image
    if len(processed_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(processed_img)
    
    # Get detailed OCR data
    custom_config = r'--oem 1 --psm 6'
    data = pytesseract.image_to_data(pil_img, config=custom_config, output_type=pytesseract.Output.DICT)

    # Collect all words with their positions
    words = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            words.append({
                'text': data['text'][i].strip(),
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'block_num': data['block_num'][i],
                'line_num': data['line_num'][i],
                'conf': data['conf'][i]
            })

    if not words:
        print("No text detected.")
        return None

    # Filter words based on start position
    if start_line is not None:
        print(f"Filtering to include only line {start_line} onwards...")
        words = [w for w in words if w['line_num'] >= start_line]
        print(f"Remaining words after filtering: {len(words)}")
    
    if start_y_position is not None:
        print(f"Filtering to include only Y-position >= {start_y_position}...")
        words = [w for w in words if w['top'] >= start_y_position]
        print(f"Remaining words after filtering: {len(words)}")

    if not words:
        print("Warning: No words found after filtering!")
        return None

    print(f"\nProcessing {len(words)} words with {len(x_positions)} column dividers")
    print(f"Column dividers at x-positions: {x_positions}")
    
    # Sort words by y position (row-wise), then x position
    words.sort(key=lambda w: (w['top'], w['left']))

    # Define column boundaries
    # Column 0: x < x_positions[0] (before 1st line)
    # Column 1: x_positions[0] <= x < x_positions[1] (between 1st and 2nd line)
    # Column 2: x_positions[1] <= x < x_positions[2] (between 2nd and 3rd line)
    # ... and so on
    # Last column: x >= x_positions[-1] (after last line)
    
    def get_column_index(word_center_x, x_positions):
        """Determine which column a word belongs to based on its center x-coordinate"""
        for i, x_line in enumerate(x_positions):
            if word_center_x < x_line:
                return i
        return len(x_positions)  # After the last line
    
    # Group words by rows
    row_threshold = 20  # Pixels to consider same row
    rows = []
    current_row_y = words[0]['top']
    current_row = []

    for word in words:
        if abs(word['top'] - current_row_y) < row_threshold:
            current_row.append(word)
        else:
            rows.append(current_row)
            current_row = [word]
            current_row_y = word['top']
    if current_row:
        rows.append(current_row)

    print(f"Grouped into {len(rows)} rows")
    
    # Prepare structured table with proper column assignment
    num_cols = len(census_headers)
    table_data = []

    for row_idx, row in enumerate(rows):
        row_cells = [''] * num_cols
        
        for word in row:
            # Calculate word center x-coordinate
            word_center = word['left'] + word['width'] / 2
            
            # Determine column index
            col_idx = get_column_index(word_center, x_positions)
            
            # Ensure column index is within bounds
            if col_idx < num_cols:
                if row_cells[col_idx]:
                    row_cells[col_idx] += ' '
                row_cells[col_idx] += word['text']
        
        table_data.append([cell.strip() for cell in row_cells])

    # Convert to DataFrame with census headers
    df = pd.DataFrame(table_data, columns=census_headers)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Structured CSV saved to: {output_csv}")
    
    # Show statistics
    print("\nColumn statistics:")
    for header in census_headers:
        non_empty = df[header].astype(str).str.strip().ne('').sum()
        print(f"  {header}: {non_empty} non-empty entries")
    
    # Save formatted table
    table_dict = {header: df[header].tolist() for header in census_headers}
    save_table_as_text(table_dict, output_txt)
    
    # Show preview
    print("\nPreview (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    return df

def smart_table_extraction_pipeline(image_path, detailed_lines, 
                                    preprocessing='original', 
                                    census_headers=None, start_line=12, 
                                    start_y_position=None,
                                    min_line_length=400):
    """
    Complete pipeline for smart table extraction using vertical line detection
    and proper column alignment based on headers
    
    Parameters:
    - image_path: Path to the input image
    - preprocessing: Preprocessing method ('original', 'grayscale', 'contrast', etc.)
    - census_headers: List of column headers (e.g., ['No.', 'Name of Children', ...])
    - start_line: Optional line number to start extraction from
    - start_y_position: Optional Y-coordinate to start extraction from
    - min_line_length: Minimum length for vertical lines to be considered dividers
    
    Returns:
    - DataFrame with extracted table data
    """
    print("="*80)
    print("SMART TABLE EXTRACTION PIPELINE")
    print("="*80)
    
    # Step 1: Preprocess the image
    print("\nSTEP 1: Preprocessing image...")
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Save preprocessed image temporarily for line detection
    temp_path = 'temp_preprocessed_for_lines.jpg'
    cv2.imwrite(temp_path, processed_img)
    print(f"✓ Preprocessed image saved temporarily")
    
    # Step 2: Detect vertical lines
    print("\nSTEP 2: Detecting vertical lines (column dividers)...")
    x_positions = detect_vertical_lines(image_path=temp_path, min_length=min_line_length)
    
    # Clean up temp file
    # try:
    #     os.remove(temp_path)
    # except:
    #     pass
    
    if not x_positions:
        print("❌ No vertical lines detected!")
        print("   Tip: Try adjusting min_line_length parameter or check image quality")
        return None
    
    # Step 3: Validate detected lines match expected columns
    if census_headers:
        expected_dividers = len(census_headers) - 1  # n columns need n-1 dividers
        print(f"\nExpected {expected_dividers} dividers for {len(census_headers)} columns")
        print(f"Detected {len(x_positions)} dividers")
        
        if len(x_positions) < expected_dividers:
            print(f"⚠️  Warning: Fewer dividers detected than expected!")
            print(f"   This may result in merged columns")
        elif len(x_positions) > expected_dividers:
            print(f"⚠️  Warning: More dividers detected than expected!")
            print(f"   Will use the {expected_dividers} best-spaced dividers")
            # Keep only the most relevant dividers
            x_positions = x_positions[:expected_dividers]

    # Step 3: Extract text and align by columns
    print("\nSTEP 3: Extracting text and assigning to columns...")
    print(f"Column assignment:")
    if census_headers:
        print(f"  Text before x={x_positions[0] if x_positions else 'N/A'} → {census_headers[0]}")
        for i in range(len(x_positions) - 1):
            print(f"  Text between x={x_positions[i]} and x={x_positions[i+1]} → {census_headers[i+1]}")
        if len(census_headers) > len(x_positions):
            print(f"  Text after x={x_positions[-1] if x_positions else 'N/A'} → {census_headers[-1]}")
    
    # df = extract_text_by_columns_with_headers(
    #     image_path, 
    #     detailed_lines
    #     x_positions, 
    #     census_headers,
    #     output_txt="structured_table.txt",
    #     output_csv="structured_table.csv",
    #     preprocessing=preprocessing,
    #     start_line=start_line,
    #     start_y_position=start_y_position
    # )
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)

    return df


def save_table_as_text(table_data, filename='table_output.txt'):
    """
    Save table data in a readable text format with columns aligned
    
    Parameters:
    - table_data: Dictionary with column headers and data
    - filename: Output filename
    """
    if not table_data:
        print("No table data to save.")
        return
    
    # Calculate column widths
    col_widths = {}
    for header, values in table_data.items():
        max_width = len(header)
        for value in values:
            max_width = max(max_width, len(str(value)))
        col_widths[header] = min(max_width + 2, 50)  # Cap at 50 chars
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Write headers
        header_line = ""
        separator_line = ""
        for header in table_data.keys():
            width = col_widths[header]
            header_line += header.ljust(width) + "| "
            separator_line += "-" * width + "+-"
        
        f.write(header_line + "\n")
        f.write(separator_line + "\n")
        
        # Write data rows
        num_rows = len(next(iter(table_data.values())))
        for i in range(num_rows):
            row_line = ""
            for header in table_data.keys():
                value = table_data[header][i] if i < len(table_data[header]) else ''
                width = col_widths[header]
                # Truncate if too long
                if len(value) > width - 2:
                    value = value[:width-5] + "..."
                row_line += value.ljust(width) + "| "
            f.write(row_line + "\n")
    
    print(f"✓ Formatted table saved to '{filename}'")

def classify_word_type(text):
    """
    Classify a word to determine which column it likely belongs to
    
    Returns:
    - 'number': Sequential number (No.)
    - 'name': Text/name (Name of Children or Name of Parent)
    - 'age': Number that looks like age
    - 'address': Mixed alphanumeric (Address)
    """
    text = text.strip()
    
    if not text:
        return 'empty'
    
    # Check if it's a simple number (1-2 digits) - likely row number
    if text.isdigit() and len(text) <= 2:
        return 'number'
    
    # Check if it contains digits and slashes or common address patterns
    # Examples: "10/707", "418", "Rt.", "St.", "Ave."
    if any(pattern in text.lower() for pattern in ['/','rt.', 'st.', 'ave.', 'rd.', 'blvd.']):
        return 'address'
    
    # Check if it's a number that could be age or part of address
    if text.isdigit():
        num = int(text)
        if 5 <= num <= 21:  # Age range for school census
            return 'age'
        else:
            return 'address'  # Likely house number
    
    # Check if it's mostly alphabetic - likely a name
    if text.replace('.', '').replace(',', '').isalpha():
        return 'name'
    
    # Mixed alphanumeric - could be address or name with initials
    alpha_count = sum(c.isalpha() for c in text)
    digit_count = sum(c.isdigit() for c in text)
    
    if digit_count > alpha_count:
        return 'address'
    else:
        return 'name'

def extract_census_columns(image_path, preprocessing='original', 
                           start_line=12, start_y_position=389):
    """
    Specialized function for census document column extraction
    Uses known census table structure and starts from specified line
    
    Parameters:
    - image_path: Path to census image
    - preprocessing: Preprocessing method
    - start_line: Line number to start from (default: 12)
    - start_y_position: Y-coordinate to start from (default: 389)
    """
    # Common census column headers
    census_headers = [
        'No.',
        'Name of Children', 
        'Age',
        'Address',
        'Name of Parent or Guardian'
    ]
    
    print(f"\nExtracting census data starting from Line {start_line} (Y={start_y_position})...")
    print(f"Headers: {census_headers}")
    
    table_data = extract_table_with_columns(
        image_path, 
        preprocessing=preprocessing,
        manual_headers=census_headers,
        save_csv=True,
        start_line=start_line,
        start_y_position=start_y_position
    )
    
    return table_data

def smart_extract_table_by_pattern(image_path, preprocessing='original',
                                   start_line=12, start_y_position=389,
                                   save_csv=True):
    """
    Extract table using intelligent pattern recognition
    Assigns words to columns based on content pattern:
    1. Number -> No. column
    2. Text/Letters -> Name of Children column
    3. Number (age range) -> Age column
    4. Mixed/Address pattern -> Address column
    5. Text (remaining) -> Name of Parent/Guardian column
    
    Parameters:
    - image_path: Path to image
    - preprocessing: Preprocessing method to use
    - start_line: Line number to start table extraction from
    - start_y_position: Y-coordinate to start from
    - save_csv: Whether to save as CSV file
    
    Returns:
    - Dictionary with column data
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Convert to PIL Image
    if len(processed_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(processed_img)
    
    custom_config = r'--oem 1 --psm 6'
    
    # Get detailed data
    data = pytesseract.image_to_data(
        pil_img, 
        config=custom_config, 
        output_type=pytesseract.Output.DICT
    )

    # Collect all words with their positions
    words = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            words.append({
                'text': data['text'][i].strip(),
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': data['conf'][i],
                'block_num': data['block_num'][i],
                'line_num': data['line_num'][i]
            })
    
    if not words:
        return {}
    
    # Filter words based on start position
    if start_line is not None:
        words = [w for w in words if w['line_num'] >= start_line]
    
    if start_y_position is not None:
        words = [w for w in words if w['top'] >= start_y_position]
    
    if not words:
        print("Warning: No words found after filtering!")
        return {}
    
    print(f"Processing {len(words)} words with pattern recognition...")
    
    # Sort words by position (top to bottom, left to right)
    words.sort(key=lambda w: (w['top'], w['left']))
    
    # Column headers
    headers = ['No.', 'Name of Children', 'Age', 'Address', 'Name of Parent or Guardian']
    
    # Initialize data structure
    table_data = {header: [] for header in headers}
    
    # Process words row by row
    current_row = {header: [] for header in headers}
    current_y = words[0]['top'] if words else 0
    row_threshold = 20  # Pixels to consider same row
    
    for word in words:
        # Check if we're on a new row
        if abs(word['top'] - current_y) > row_threshold:
            # Save current row
            for header in headers:
                table_data[header].append(' '.join(current_row[header]))
            
            # Reset for new row
            current_row = {header: [] for header in headers}
            current_y = word['top']
        
        # Classify word and assign to appropriate column
        word_type = classify_word_type(word['text'])
        
        # Determine which column based on pattern and what's already filled
        if word_type == 'number' and not current_row['No.']:
            # First number in row -> No. column
            current_row['No.'].append(word['text'])
        
        elif word_type == 'name':
            # Name-like text
            if not current_row['Name of Children']:
                # First name -> Name of Children
                current_row['Name of Children'].append(word['text'])
            else:
                # Subsequent names -> Name of Parent/Guardian
                current_row['Name of Parent or Guardian'].append(word['text'])
        
        elif word_type == 'age':
            # Age number
            if not current_row['Age']:
                current_row['Age'].append(word['text'])
            else:
                # If age already filled, might be part of address
                current_row['Address'].append(word['text'])
        
        elif word_type == 'address':
            # Address-like text
            current_row['Address'].append(word['text'])
        
        else:
            # Fallback: add to Name of Children if empty, else Parent/Guardian
            if not current_row['Name of Children']:
                current_row['Name of Children'].append(word['text'])
            else:
                current_row['Name of Parent or Guardian'].append(word['text'])
    
    # Add last row
    if any(current_row.values()):
        for header in headers:
            table_data[header].append(' '.join(current_row[header]))
    
    # Equalize column lengths
    max_length = max(len(col_data) for col_data in table_data.values()) if table_data else 0
    for header in table_data:
        while len(table_data[header]) < max_length:
            table_data[header].append('')
    
    print(f"✓ Extracted {max_length} rows using pattern recognition")
    
    # Show classification statistics
    print("\nPattern recognition results:")
    for header, values in table_data.items():
        non_empty = sum(1 for v in values if v.strip())
        print(f"  {header}: {non_empty} non-empty entries")
    
    # Save as CSV if requested
    if save_csv:
        df = pd.DataFrame(table_data)
        df.to_csv('census_smart_extraction.csv', index=False)
        print(f"\n✓ Smart extraction saved to 'census_smart_extraction.csv'")
    
    return table_data

def extract_census_smart(image_path, preprocessing='original',
                         start_line=12, start_y_position=389):
    """
    Smart census extraction using pattern recognition
    Wrapper function for easy use
    """
    print(f"\n{'='*80}")
    print("SMART CENSUS EXTRACTION (Pattern-Based)")
    print(f"{'='*80}")
    print(f"Starting from Line {start_line} (Y={start_y_position})")
    print("\nPattern rules:")
    print("  1. First number → No. column")
    print("  2. First text/name → Name of Children column")
    print("  3. Age number (5-21) → Age column")
    print("  4. Address pattern (numbers, /, St., Rt.) → Address column")
    print("  5. Remaining text → Name of Parent/Guardian column")
    print(f"{'='*80}\n")
    
    table_data = smart_extract_table_by_pattern(
        image_path,
        preprocessing=preprocessing,
        start_line=start_line,
        start_y_position=start_y_position,
        save_csv=True
    )
    
    if table_data:
        # Save as formatted text
        save_table_as_text(table_data, 'census_smart_formatted.txt')
    
    return table_data

# Example usage
if __name__ == "__main__":
    image_path = "img.jpg"  # Replace with your image path
    
    print("=" * 80)
    print("OCR EXTRACTION WITH MULTIPLE PREPROCESSING OPTIONS")
    print("=" * 80)
    
    print("\nSTEP 1: Testing preprocessing methods...")
    print("-" * 80)
    
    # Test different preprocessing methods
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
            _, processed = preprocess_image(image_path, method=method, 
                                           save_steps=(method in ['contrast', 'advanced']))
            cv2.imwrite(f'preprocessed_{method}.jpg', processed)
            print(f"    ✓ Saved: preprocessed_{method}.jpg")
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:50]}")
    
    print("\n✓ Review these images to see the preprocessing effects!")
    
    print("\n" + "=" * 80)
    print("STEP 2: Testing OCR with different preprocessing methods...")
    print("=" * 80)
    
    # Test PSM modes with different preprocessing
    psm_configs = [
        ('--psm 4 --oem 1', 'PSM 4: Single column + LSTM'),
        ('--psm 6 --oem 1', 'PSM 6: Uniform block + LSTM'),
        ('--psm 3 --oem 1', 'PSM 3: Fully automatic + LSTM'),
    ]
    
    # Test preprocessing methods
    test_preprocessing = ['original', 'grayscale', 'contrast', 'bilateral', 'advanced']
    
    print("\nTesting combinations of PSM modes and preprocessing...")
    print("-" * 80)
    
    results = []
    
    for config, config_desc in psm_configs:
        print(f"\n{config_desc}:")
        for preproc in test_preprocessing:
            try:
                data, img_boxes, words = extract_text_with_boxes(
                    image_path, 
                    f'output_{config.replace(" ", "_").replace("--", "")}_{preproc}.jpg',
                    config=config,
                    min_conf=0,
                    preprocessing=preproc
                )
                print(f"  {preproc:20s}: {len(words):4d} words detected")
                results.append({
                    'config': config,
                    'preproc': preproc,
                    'count': len(words),
                    'description': config_desc
                })
            except Exception as e:
                print(f"  {preproc:20s}: Error - {str(e)[:50]}")
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x['count'])
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION FOUND:")
        print("=" * 80)
        print(f"  Tesseract config: {best['config']}")
        print(f"  Preprocessing: {best['preproc']}")
        print(f"  Words detected: {best['count']}")
        print("=" * 80)
        
        # Extract text with best configuration
        print("\nSTEP 3: Extracting text with best configuration...")
        print("-" * 80)
        
        print("\nExtracting line-by-line text...")
        line_text = extract_text_by_lines(
            image_path, 
            config=best['config'], 
            min_conf=0,
            preprocessing=best['preproc']
        )
        
        # Show preview
        print("\nFirst 40 lines of extracted text:")
        print("-" * 80)
        lines_preview = line_text.split('\n')[:40]
        for i, line in enumerate(lines_preview, 1):
            if line.strip():
                print(f"{i:3d}: {line}")
        print("-" * 80)
        
        # Save results
        with open('ocr_text_lines.txt', 'w', encoding='utf-8') as f:
            f.write(line_text)
        print("\n✓ Full text saved to 'ocr_text_lines.txt'")
        
        # Extract with confidence filtering
        print("\nExtracting with confidence threshold (min_conf=30)...")
        line_text_filtered = extract_text_by_lines(
            image_path, 
            config=best['config'], 
            min_conf=30,
            preprocessing=best['preproc']
        )
        
        with open('ocr_text_lines_filtered.txt', 'w', encoding='utf-8') as f:
            f.write(line_text_filtered)
        print("✓ Filtered text saved to 'ocr_text_lines_filtered.txt'")
        
        # Get detailed line information
        print("\nExtracting detailed line information...")
        detailed_lines = extract_text_by_lines_detailed(
            image_path, 
            config=best['config'], 
            min_conf=0,
            preprocessing=best['preproc']
        )
        
        print(f"✓ Found {len(detailed_lines)} lines")
        
        # Save detailed information
        with open('ocr_lines_detailed.txt', 'w', encoding='utf-8') as f:
            f.write("Detailed Line Information:\n")
            f.write(f"Best Config: {best['config']} with {best['preproc']} preprocessing\n")
            f.write("=" * 80 + "\n\n")
            for i, line in enumerate(detailed_lines, 1):
                f.write(f"Line {i} (Block {line['block']}, Line {line['line']}):\n")
                f.write(f"Position: ({line['left']:.1f}, {line['top']:.1f})\n")
                f.write(f"Text: {line['text']}\n")
                f.write(f"Words: {len(line['words'])}\n")
                for word in line['words']:
                    f.write(f"  '{word['text']}' | pos:({word['left']},{word['top']}) | conf:{word['conf']:.1f}\n")
                f.write("\n")
        
        print("✓ Detailed info saved to 'ocr_lines_detailed.txt'")
        
        # STEP 4: Additional extraction methods
        print("\n" + "=" * 80)
        print("STEP 4: Advanced extraction methods...")
        print("=" * 80)
        
        # Extract with confidence scores
        print("\n4a. Extracting with confidence scores...")
        conf_results = extract_text_with_confidence(image_path, preprocessing=best['preproc'])
        print(f"✓ Found {len(conf_results)} words with confidence > 0")
        print("\nTop 10 high confidence words:")
        sorted_results = sorted(conf_results, key=lambda x: x['confidence'], reverse=True)[:10]
        for r in sorted_results:
            print(f"  '{r['text']}' (confidence: {r['confidence']}%)")
        
        # Extract structured data
        print("\n4b. Extracting table with columns...")
        
        # Option 1: Auto-detect columns
        print("\n  Method 1: Auto-detect column structure")
        table_auto = extract_table_with_columns(
            image_path, 
            preprocessing=best['preproc'],
            manual_headers=None,
            save_csv=True
        )
        
        if table_auto:
            print(f"  ✓ Detected {len(table_auto)} columns")
            print("  Column headers:", list(table_auto.keys()))
            
            # Show preview
            print("\n  Preview of extracted table (first 5 rows):")
            for header in table_auto.keys():
                print(f"    {header}: {len(table_auto[header])} entries")
            
            # Save as formatted text
            save_table_as_text(table_auto, 'table_auto_detected.txt')
        
        # Option 2: Use manual headers (for census documents)
        print("\n  Method 2: Using predefined census headers (from line 12)")
        census_headers = [
            'No.',
            'Name of Children',
            'Age',
            'Address', 
            'Name of Parent or Guardian'
        ]
        
        # Start from line 12, Y-position 389 (as per your requirement)
        table_manual = extract_table_with_columns(
            image_path,
            preprocessing=best['preproc'],
            manual_headers=census_headers,
            save_csv=False,
            start_line=12,
            start_y_position=389
        )
        
        if table_manual:
            # Save with custom filename
            df_manual = pd.DataFrame(table_manual)
            df_manual.to_csv('census_table_columns.csv', index=False)
            print(f"  ✓ Census table saved to 'census_table_columns.csv'")
            
            # Save as formatted text
            save_table_as_text(table_manual, 'census_table_formatted.txt')
            
            # Show statistics
            print(f"\n  Column statistics:")
            for header, values in table_manual.items():
                non_empty = sum(1 for v in values if v.strip())
                print(f"    {header}: {non_empty} non-empty entries")
        
        # Option 3: Use specialized census extraction
        print("\n  Method 3: Specialized census extraction (from line 12)")
        census_data = extract_census_columns(
            image_path, 
            preprocessing=best['preproc'],
            start_line=12,
            start_y_position=389
        )
        
        if census_data:
            save_table_as_text(census_data, 'census_specialized.txt')
            print("  ✓ Specialized census extraction complete")
        
        # Option 4: SMART PATTERN-BASED EXTRACTION (NEW!)
        print("\n  Method 4: Smart pattern-based extraction (RECOMMENDED)")
        print("  This method intelligently assigns words to columns based on content:")
        print("    - Numbers (1-2 digits) → No. column")
        print("    - First text → Name of Children")
        print("    - Age numbers (5-21) → Age column")
        print("    - Address patterns → Address column")
        print("    - Remaining text → Parent/Guardian column")
        
        smart_data = extract_census_smart(
            image_path,
            preprocessing=best['preproc'],
            start_line=12,
            start_y_position=389
        )
        if smart_data:
            save_table_as_text(smart_data, 'census_smart_specialized.txt')
            print("  ✓ Specialized census extraction complete")
        
        # Option 5: SMART VERTICAL LINE-BASED EXTRACTION (BEST METHOD!)
        print("\n  Method 5: Vertical line-based extraction with column headers (BEST!)")
        print("  This method:")
        print("    - Detects vertical lines that separate columns")
        print("    - Assigns text before 1st line → No. column")
        print("    - Assigns text between lines → respective columns")
        print("    - Most accurate for table documents with visible borders")
        
        census_headers = [
            'No.',
            'Name of Children',
            'Age',
            'Address', 
            'Name of Parent or Guardian'
        ]
        
        # df_vertical = smart_table_extraction_pipeline(
        #     image_path,
        #     detailed_lines = detailed_lines
        #     preprocessing=best['preproc'],
        #     census_headers=census_headers,
        #     start_line=12,
        #     start_y_position=389,
        #     min_line_length=500  # Adjust if needed
        # )

        x_positions = detect_vertical_lines(image_path=image_path, min_length=500)

        table = assign_text_to_columns(detailed_lines, x_positions, census_headers)

        save_table_to_markdown(table)

        # if df_vertical is not None:

        #     df_vertical.to_string('census_vertical_specialized.txt', index=False)

        #     # save_table_as_text(census_data, 'census_vertical_specialized.txt')
        #     print("  ✓ Specialized census extraction complete")
        
        # if df_vertical is not None:
        #     print("  ✓ Vertical line-based extraction complete!")
        #     print(f"  ✓ Extracted {len(df_vertical)} rows with proper column alignment")
        
        # else:
        #     print("  ✗ Vertical line detection failed. Try adjusting min_line_length parameter.")

        # #TODO: Implement this to create a file
        # for i, line in enumerate(detailed_lines, 1):
        #         f.write(f"Line {i} (Block {line['block']}, Line {line['line']}):\n")
        #         f.write(f"Position: ({line['left']:.1f}, {line['top']:.1f})\n")
        #         f.write(f"Text: {line['text']}\n")
        #         f.write(f"Words: {len(line['words'])}\n")
        #         for word in line['words']:
        #             f.write(f"  '{word['text']}' | pos:({word['left']},{word['top']}) | conf:{word['conf']:.1f}\n")
        #         f.write("\n")

        
        print("\n" + "=" * 80)
        print("SUMMARY OF OUTPUT FILES:")
        print("=" * 80)
        
        print("\n1. PREPROCESSING COMPARISON:")
        print("   View these to understand each preprocessing method:")
        for method, desc in preprocessing_methods:
            print(f"   • preprocessed_{method}.jpg - {desc}")
        
        print("\n2. OCR OUTPUT WITH BOUNDING BOXES:")
        print("   • output_psm*_*.jpg - Visual verification of text detection")
        print("   • Check the one matching best config for accuracy")
        
        print("\n3. EXTRACTED TEXT:")
        print("   • ocr_text_lines.txt - All extracted text (line by line)")
        print("   • ocr_text_lines_filtered.txt - High confidence text only")
        print("   • ocr_lines_detailed.txt - Detailed metadata and positions")
        
        print("\n4. TABLE/COLUMN EXTRACTION:")
        print("   • table_columns.csv - Auto-detected table structure")
        print("   • census_table_columns.csv - Census with predefined headers")
        print("   • census_smart_extraction.csv - SMART pattern-based extraction (BEST)")
        print("   • table_auto_detected.txt - Formatted table (auto headers)")
        print("   • census_table_formatted.txt - Formatted census table")
        print("   • census_smart_formatted.txt - SMART pattern-based formatted (BEST)")
        print("   • census_specialized.txt - Specialized census extraction")
        
        print("\n5. ADDITIONAL FILES (if using bilateral/advanced preprocessing):")
        print("   • out/processed_image.jpg - ImagePreprocessor output")
        print("   • step*.jpg - Intermediate processing steps")
        
        print("\n" + "=" * 80)
        print("PREPROCESSING METHOD DESCRIPTIONS:")
        print("=" * 80)
        print("• original: No processing (best for high-quality scans)")
        print("• grayscale: Simple black & white conversion")
        print("• contrast: CLAHE enhancement (makes faint text clearer)")
        print("• bilateral: Bilateral filter + adaptive threshold (edge-preserving)")
        print("• advanced: 2x upscale + denoise + CLAHE + Otsu (comprehensive)")
        print("• otsu: Binary threshold (automatic threshold selection)")
        print("• adaptive: Adaptive threshold (good for uneven lighting)")
        
        print("\n" + "=" * 80)
        print("HOW TO USE THE TABLE EXTRACTION:")
        print("=" * 80)
        print("\n1. AUTO-DETECT COLUMNS:")
        print("   table_data = extract_table_with_columns('image.jpg')")
        print("   • Automatically detects column boundaries")
        print("   • Tries to extract headers from image")
        
        print("\n2. MANUAL HEADERS (RECOMMENDED FOR YOUR CENSUS):")
        print("   headers = ['No.', 'Name', 'Age', 'Address', 'Parent']")
        print("   table_data = extract_table_with_columns(")
        print("       'image.jpg',")
        print("       manual_headers=headers,")
        print("       start_line=12,          # Start from line 12")
        print("       start_y_position=389    # Or Y-position 389")
        print("   )")
        print("   • Uses your specified column headers")
        print("   • Starts extraction from specified line/position")
        print("   • Better accuracy for known table structures")
        
        print("\n3. CENSUS EXTRACTION (EASIEST):")
        print("   census_data = extract_census_columns(")
        print("       'image.jpg',")
        print("       start_line=12,")
        print("       start_y_position=389")
        print("   )")
        print("   • Pre-configured for census documents")
        print("   • Automatic column headers and structure")
        print("   • Starts from line 12 (Y=389) by default")
        
        print("\n4. SMART PATTERN-BASED EXTRACTION (BEST FOR YOUR CASE):")
        print("   smart_data = extract_census_smart(")
        print("       'image.jpg',")
        print("       start_line=12,")
        print("       start_y_position=389")
        print("   )")
        print("   • Intelligently assigns words to correct columns")
        print("   • Pattern recognition: Number→No., Text→Name, Age→Age, etc.")
        print("   • No need to specify column positions")
        print("   • RECOMMENDED for your requirements!")
        
        print("\n5. CUSTOMIZE START POSITION:")
        print("   • Check 'ocr_lines_detailed.txt' to find line numbers")
        print("   • Use start_line parameter for line-based filtering")
        print("   • Use start_y_position parameter for Y-coordinate filtering")
        print("   • Both parameters can be used together for precision")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        print("✓ Compare preprocessed images to see which looks clearest")
        print("✓ Check output images to verify text detection accuracy")
        print("✓ Review census_table_formatted.txt for readable column data")
        print("✓ Open census_table_columns.csv in Excel for easy viewing")
        print("✓ For high-quality scans, 'original' or 'grayscale' often works best")
        print("✓ For faint/old documents, try 'bilateral' or 'advanced'")
        print("✓ PSM 4 is usually best for column-based documents like census forms")
        print("✓ If columns are misaligned, adjust manual_headers in the code")
        print("=" * 80)
        
    else:
        print("\n✗ No successful OCR extractions.")
        print("  Please check:")
        print("  1. Tesseract is installed: 'tesseract --version'")
        print("  2. Image path is correct")
        print("  3. Image is readable by OpenCV")
        print("  4. pandas is installed: 'pip install pandas'")