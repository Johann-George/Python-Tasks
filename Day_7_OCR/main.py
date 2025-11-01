import cv2
import pytesseract
from PIL import Image
import numpy as np

def preprocess_image(image_path, method='adaptive'):
    """
    Preprocess the image for better OCR results
    method: 'adaptive', 'otsu', 'simple', or 'none'
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'adaptive':
        # Adaptive thresholding works better for uneven lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    elif method == 'otsu':
        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'simple':
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        thresh = gray
    
    # Optional: Denoise (can be slow)
    # denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Increase contrast
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return img, processed

def extract_text_with_boxes(image_path, output_image_path='output_with_boxes.jpg', 
                           config='--psm 6', min_conf=0):
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
    # Try different preprocessing methods
    original_img = cv2.imread(image_path)
    
    # Test multiple preprocessing approaches
    methods = ['adaptive', 'otsu', 'none']
    best_count = 0
    best_processed = None
    
    for method in methods:
        _, processed = preprocess_image(image_path, method)
        # Quick test to see which method detects more text
        test_data = pytesseract.image_to_data(processed, config=config, 
                                              output_type=pytesseract.Output.DICT)
        count = sum(1 for conf in test_data['conf'] if int(conf) > min_conf)
        if count > best_count:
            best_count = count
            best_processed = processed
    
    # Use the best preprocessing method
    processed_img = best_processed
    
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

def extract_text_by_lines(image_path, config='--psm 6', min_conf=0):
    """
    Extract text organized by lines, preserving the original layout
    """
    _, processed_img = preprocess_image(image_path, method='adaptive')
    
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

def extract_text_by_lines_detailed(image_path, config='--psm 6', min_conf=0):
    """
    Extract text organized by lines with detailed information
    Returns list of lines with their positions and words
    """
    _, processed_img = preprocess_image(image_path, method='adaptive')
    
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

def get_detailed_data(image_path, config='--psm 6', min_conf=0):
    """
    Get detailed OCR data including confidence scores
    """
    _, processed_img = preprocess_image(image_path, method='adaptive')
    
    # Get detailed data
    data = pytesseract.image_to_data(processed_img, config=config,
                                     output_type=pytesseract.Output.DICT)
    
    # Create a structured output
    results = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        if int(data['conf'][i]) > min_conf:
            text = data['text'][i].strip()
            if text:
                results.append({
                    'text': text,
                    'confidence': data['conf'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'block_num': data['block_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i]
                })
    
    return results

# Example usage
if __name__ == "__main__":
    image_path = "img.jpg"  # Replace with your image path
    
    print("=" * 80)
    print("Testing different PSM modes for better extraction...")
    print("=" * 80)
    
    # Test different PSM modes
    psm_modes = [
        ('--psm 6', 'Single uniform block of text (default)'),
        ('--psm 4', 'Single column of text'),
        ('--psm 3', 'Fully automatic page segmentation'),
        ('--psm 11', 'Sparse text - finds as much as possible'),
    ]
    
    best_mode = None
    best_count = 0
    
    for config, description in psm_modes:
        print(f"\nTrying {description}...")
        try:
            data, img_boxes, words = extract_text_with_boxes(
                image_path, 
                f'output_{config.split()[1]}.jpg',
                config=config,
                min_conf=0  # Accept all detections
            )
            if len(words) > best_count:
                best_count = len(words)
                best_mode = config
        except Exception as e:
            print(f"Error with {config}: {e}")
    
    print("\n" + "=" * 80)
    print(f"Best mode: {best_mode} with {best_count} detected words")
    print("=" * 80)
    
    # Extract text line by line with best mode
    print("\nExtracting text LINE BY LINE with best configuration...")
    line_text = extract_text_by_lines(image_path, config=best_mode, min_conf=0)
    print("\nExtracted Text (Line by Line):")
    print("-" * 80)
    print(line_text)
    print("-" * 80)
    
    # Save line-by-line text
    with open('ocr_text_lines.txt', 'w', encoding='utf-8') as f:
        f.write(line_text)
    print("\nLine-by-line text saved to 'ocr_text_lines.txt'")
    
    # Get detailed line information
    print("\nExtracting detailed line information...")
    detailed_lines = extract_text_by_lines_detailed(image_path, config=best_mode, min_conf=0)
    
    print(f"\nFound {len(detailed_lines)} lines of text")
    print("\nFirst 10 lines:")
    for i, line in enumerate(detailed_lines[:10], 1):
        print(f"Line {i}: {line['text']}")
    
    # Save detailed line information
    with open('ocr_lines_detailed.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Line Information:\n")
        f.write("=" * 80 + "\n\n")
        for i, line in enumerate(detailed_lines, 1):
            f.write(f"Line {i} (Block {line['block']}, Line {line['line']}):\n")
            f.write(f"Position: ({line['left']:.1f}, {line['top']:.1f})\n")
            f.write(f"Text: {line['text']}\n")
            f.write(f"Words ({len(line['words'])}):\n")
            for word in line['words']:
                f.write(f"  - '{word['text']}' at ({word['left']}, {word['top']}) "
                       f"conf: {word['conf']:.1f}\n")
            f.write("\n")
    
    print("Detailed line information saved to 'ocr_lines_detailed.txt'")
    
    # Get detailed word data (original functionality)
    print("\nGetting detailed OCR data with best configuration...")
    detailed_results = get_detailed_data(image_path, config=best_mode, min_conf=0)
    
    print(f"\nFound {len(detailed_results)} words/tokens")
    
    # Save detailed results to file
    with open('ocr_results.txt', 'w', encoding='utf-8') as f:
        f.write("All Detected Text:\n")
        f.write("=" * 80 + "\n")
        for result in detailed_results:
            f.write(f"Text: '{result['text']}' | "
                   f"Conf: {result['confidence']:.1f} | "
                   f"Pos: ({result['left']}, {result['top']}) | "
                   f"Size: {result['width']}x{result['height']}\n")
    
    print("\nDetailed word results saved to 'ocr_results.txt'")
    
    # Display sample results
    print("\nFirst 20 words with details:")
    for result in detailed_results[:20]:
        print(f"Text: '{result['text']}', "
              f"Confidence: {result['confidence']:.1f}, "
              f"Position: ({result['left']}, {result['top']}), "
              f"Size: {result['width']}x{result['height']}")
