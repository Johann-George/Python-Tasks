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

def extract_structured_data(image_path, preprocessing='original', min_conf=30):
    """
    Extract structured data (useful for tables)
    Returns pandas DataFrame
    """
    _, processed_img = preprocess_image(image_path, method=preprocessing)
    
    # Convert to PIL Image
    if len(processed_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(processed_img)
    
    custom_config = r'--oem 3 --psm 6'
    
    data = pytesseract.image_to_data(
        pil_img, 
        config=custom_config, 
        output_type=pytesseract.Output.DICT
    )
    
    df = pd.DataFrame(data)
    
    # Filter out empty text and low confidence
    df = df[df['conf'] > min_conf]
    df = df[df['text'].str.strip() != '']
    
    return df

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
        print("\n4b. Extracting structured data (DataFrame)...")
        df = extract_structured_data(image_path, preprocessing=best['preproc'], min_conf=30)
        print(f"✓ DataFrame shape: {df.shape}")
        print("\nFirst 10 rows:")
        print(df[['text', 'conf', 'left', 'top', 'width', 'height']].head(10).to_string())
        df.to_csv('structured_output.csv', index=False)
        print("\n✓ Saved to 'structured_output.csv'")
        
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
        print("   • structured_output.csv - Tabular data format")
        
        print("\n4. ADDITIONAL FILES (if using bilateral/advanced preprocessing):")
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
        print("RECOMMENDATIONS:")
        print("=" * 80)
        print("✓ Compare preprocessed images to see which looks clearest")
        print("✓ Check output images to verify text detection accuracy")
        print("✓ Review ocr_text_lines.txt for the extracted content")
        print("✓ For high-quality scans, 'original' or 'grayscale' often works best")
        print("✓ For faint/old documents, try 'bilateral' or 'advanced'")
        print("✓ PSM 4 is usually best for column-based documents like census forms")
        print("=" * 80)
        
    else:
        print("\n✗ No successful OCR extractions.")
        print("  Please check:")
        print("  1. Tesseract is installed: 'tesseract --version'")
        print("  2. Image path is correct")
        print("  3. Image is readable by OpenCV")
        print("  4. pandas is installed: 'pip install pandas'")