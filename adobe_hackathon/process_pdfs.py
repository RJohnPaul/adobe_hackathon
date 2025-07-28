import os
import json
import time
from pathlib import Path
import fitz
import numpy as np
from PIL import Image
import io
import cv2
import re
import yaml
import onnxruntime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

class YOLOv9:
    def __init__(self, model_path, class_mapping_path, original_size=(1280, 720), score_threshold=0.1, conf_thresold=0.4, iou_threshold=0.4, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.score_threshold = score_threshold
        self.conf_thresold = conf_thresold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = original_size
        self.create_session(model_path)

        with open(class_mapping_path, 'r') as f:
            self.classes = yaml.safe_load(f)['names']
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def create_session(self, model_path):
        providers = ['CPUExecutionProvider']
        if self.device.lower() != "cpu":
            providers.append("CUDAExecutionProvider")
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        try:
            self.input_height = int(shape[2])
            self.input_width = int(shape[3])
        except Exception:
            self.input_height, self.input_width = 1024, 1024

    def preprocess(self, img):
        resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        tensor = resized / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        return tensor

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        boxes = np.divide(boxes, [self.input_width, self.input_height, self.input_width, self.input_height])
        boxes *= [self.image_width, self.image_height, self.image_width, self.image_height]
        boxes = boxes.astype(np.int32)

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.score_threshold, self.iou_threshold)
        indices = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in indices]

        detections = []
        for i in indices:
            detections.append({
                "class_index": class_ids[i],
                "confidence": scores[i],
                "box": self.xywh2xyxy(boxes[i]),
                "class_name": self.classes[class_ids[i]]
            })
        return detections

    def detect(self, img):
        self.image_height, self.image_width = img.shape[:2]
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]
        return self.postprocess(outputs)

def load_pdf_pages(pdf_path, dpi=150):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pdf_document = fitz.open(pdf_path)
    print(f"PDF loaded: {pdf_path} ({len(pdf_document)} pages)")
    
    page_images = []
    page_objects = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_objects.append(page)
        
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=matrix)
        
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        img_array = np.array(img)
        
        page_images.append(img_array)
        print(f"Processed page {page_num+1}, shape: {img_array.shape}")
    
    return page_images, page_objects, pdf_document

def extract_text_with_font_details(page, bbox, page_num, dpi=150):
    x1, y1, x2, y2 = bbox
    
    scale_factor = 72 / dpi
    pdf_x1 = x1 * scale_factor
    pdf_y1 = y1 * scale_factor
    pdf_x2 = x2 * scale_factor
    pdf_y2 = y2 * scale_factor
    
    rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)
    
    text_dict = page.get_text("dict", clip=rect)
    
    text_info = {
        'text_content': '',
        'font_name': 'Unknown',
        'font_size': 0,
        'is_bold': False,
        'is_italic': False,
        'color': 0,
        'page_number': page_num + 1,
        'coordinates': {
            'x1': int(x1), 'y1': int(y1), 
            'x2': int(x2), 'y2': int(y2)
        },
        'pdf_coordinates': {
            'x1': pdf_x1, 'y1': pdf_y1,
            'x2': pdf_x2, 'y2': pdf_y2
        },
        'dimensions': {
            'width': int(x2 - x1),
            'height': int(y2 - y1),
            'area': int((x2 - x1) * (y2 - y1))
        }
    }
    
    all_text_parts = []
    font_sizes = []
    font_names = []
    font_flags = []
    
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if text:
                        all_text_parts.append(text)
                        font_sizes.append(span.get("size", 0))
                        font_names.append(span.get("font", "Unknown"))
                        font_flags.append(span.get("flags", 0))
    
    text_info['text_content'] = ' '.join(all_text_parts).strip()
    
    if font_sizes:
        text_info['font_size'] = max(set(font_sizes), key=font_sizes.count)
    if font_names:
        text_info['font_name'] = max(set(font_names), key=font_names.count)
    
    if font_flags:
        dominant_flags = max(set(font_flags), key=font_flags.count)
        text_info['is_bold'] = bool(dominant_flags & 2**4)
        text_info['is_italic'] = bool(dominant_flags & 2**1)
    
    return text_info

def classify_heading_level(text_info, all_headings):
    font_size = text_info['font_size']
    is_bold = text_info['is_bold']
    text_content = text_info['text_content']
    
    all_font_sizes = [h.get('font_size', 0) for h in all_headings if h.get('font_size', 0) > 0]
    
    if len(all_font_sizes) > 0:
        max_font_size = max(all_font_sizes)
        avg_font_size = sum(all_font_sizes) / len(all_font_sizes)
        
        if font_size >= max_font_size * 0.9 and is_bold:
            level = "H1"
        elif font_size >= avg_font_size * 1.1:
            level = "H2"
        else:
            level = "H3"
    else:
        if font_size >= 16 and is_bold:
            level = "H1"
        elif font_size >= 14:
            level = "H2"
        else:
            level = "H3"
    
    text_upper = text_content.upper()
    
    if (any(word in text_upper for word in ['OVERVIEW', 'INTRODUCTION', 'ABSTRACT', 'SUMMARY'])
        and is_bold and font_size >= 14):
        level = "H1"
    
    if re.match(r'^\d+\.?\s+', text_content.strip()):
        if font_size >= 14 and is_bold:
            level = "H1"
        else:
            level = "H2"
    elif re.match(r'^\d+\.\d+\.?\s+', text_content.strip()):
        level = "H2"
    elif re.match(r'^\d+\.\d+\.\d+\.?\s+', text_content.strip()):
        level = "H3"
    
    return level

def is_title_candidate(text_info, page_num, all_headings):
    text_content = text_info['text_content']
    font_size = text_info['font_size']
    is_bold = text_info['is_bold']
    coordinates = text_info['coordinates']
    
    if page_num > 2:
        return False
    
    title_keywords = ['abstract', 'introduction', 'overview', 'summary', 'conclusion',
                     'executive summary', 'table of contents', 'acknowledgments']
    
    if any(keyword in text_content.lower() for keyword in title_keywords):
        return False
    
    if len(text_content) > 100:
        return False
    
    width = coordinates['x2'] - coordinates['x1']
    if width < 200:
        return False
    
    if font_size >= 18 and is_bold:
        return True
    
    if page_num == 0 and len(all_headings) == 0 and is_bold:
        return True
    
    return False

def visualize_detections(img, detections, class_colors):
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for det in detections:
        if 'box' in det:
            x1, y1, x2, y2 = det['box'].astype(int)
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            color = class_colors.get(det['class_name'].lower(), '#AAAAAA')
        else:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det.get('class_name', 'unknown')} ({det['confidence']:.2f})"
            color = class_colors.get(det.get('class_name', 'unknown').lower(), '#AAAAAA')
        
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, label,
                bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.2'),
                fontsize=8, color='white')
    ax.axis('off')
    return fig

def detect_and_extract_headings(detector, page_images, page_objects, conf_threshold=0.25, dpi=150, save_visualizations=False, output_dir=None):
    all_headings = []
    
    layout_colors = {
        'text': '#FF5733',
        'title': '#3358FF',
        'list': '#33FF57',
        'table': '#FF33DD',
        'figure': '#9933FF',
        'caption': '#FFD633',
        'header': '#33FFF5',
        'footer': '#FF8C33',
        'page_number': '#C133FF',
        'section': '#33B5FF',
        'formula': '#B2FF33'
    }
    
    print(f"Processing {len(page_images)} pages for headings...")
    
    for i, (img, page_obj) in enumerate(zip(page_images, page_objects)):
        print(f"\nAnalyzing page {i+1}...")
        
        detections = detector.detect(img)
        print(f"Found {len(detections)} objects")
        
        page_headings = []
        
        # Only create visualization if requested
        if save_visualizations:
            fig, ax = plt.subplots(1, figsize=(12, 16))
            ax.imshow(img)
        
        for det in detections:
            class_name = det['class_name']
            
            if 'title' in class_name.lower() or 'header' in class_name.lower():
                if 'page' in class_name.lower():
                    continue
                
                x1, y1, x2, y2 = det['box'].astype(int)
                
                text_details = extract_text_with_font_details(
                    page_obj, [x1, y1, x2, y2], i, dpi
                )
                
                if not text_details['text_content'].strip():
                    continue
                
                if is_title_candidate(text_details, i, all_headings):
                    heading_level = "Title"
                else:
                    heading_level = classify_heading_level(text_details, all_headings)
                
                heading_info = {
                    'element_type': class_name,
                    'confidence': float(det['confidence']),
                    'heading_level': heading_level,
                    **text_details
                }
                
                page_headings.append(heading_info)
                all_headings.append(heading_info)
                
                # Only add visualization elements if saving visualizations
                if save_visualizations:
                    color = layout_colors.get(class_name.lower(), '#999999')
                    
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=3,
                        edgecolor=color,
                        facecolor='none',
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    text_preview = text_details['text_content'][:30] + "..." if len(text_details['text_content']) > 30 else text_details['text_content']
                    label_text = f"{heading_level}: {text_preview}\n{class_name}: {det['confidence']:.2f}"
                    
                    ax.text(
                        x1, y1-15,
                        label_text,
                        bbox=dict(facecolor=color, alpha=0.9, pad=3, boxstyle='round'),
                        fontsize=9,
                        color='white',
                        weight='bold'
                    )
                
                print(f"  {heading_level} - {class_name}:")
                print(f"     Text: '{text_details['text_content'][:80]}...'")
                print(f"     Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"     Font: {text_details['font_name']} - {text_details['font_size']:.1f}pt")
                print(f"     Style: {'Bold' if text_details['is_bold'] else 'Normal'} | {'Italic' if text_details['is_italic'] else 'Regular'}")
                print(f"     Confidence: {det['confidence']:.3f}")
                print()
        
        # Save visualization if requested, but don't show it
        if save_visualizations and output_dir:
            ax.set_title(f"Page {i+1} - Heading Detection Results")
            ax.axis('off')
            plt.tight_layout()
            
            # Save to file instead of showing
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            viz_file = viz_dir / f"page_{i+1}_headings.png"
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print(f"   📸 Visualization saved: {viz_file}")
        elif save_visualizations:
            plt.close()  # Close figure if created but no output dir
        
        print(f"  Page {i+1}: Found {len(page_headings)} headings")
    
    return all_headings

def generate_document_outline(all_headings):
    print("\nDOCUMENT OUTLINE STRUCTURE")
    print("=" * 80)
    
    sorted_headings = sorted(all_headings, key=lambda x: (x['page_number'], x['coordinates']['y1']))
    
    outline_data = []
    
    for i, heading in enumerate(sorted_headings, 1):
        outline_entry = {
            'sequence': i,
            'heading_level': heading['heading_level'],
            'text': heading['text_content'],
            'page': heading['page_number'],
            'font_size': heading['font_size'],
            'font_name': heading['font_name'],
            'is_bold': heading['is_bold'],
            'is_italic': heading['is_italic'],
            'coordinates': heading['coordinates'],
            'confidence': heading['confidence']
        }
        outline_data.append(outline_entry)
        
        indent = "  " * (int(heading['heading_level'][1]) - 1) if heading['heading_level'] != 'Title' else ""
        style_info = []
        if heading['is_bold']:
            style_info.append("Bold")
        if heading['is_italic']:
            style_info.append("Italic")
        style_str = f" ({', '.join(style_info)})" if style_info else ""
        
        print(f"{indent}{heading['heading_level']}. {heading['text_content']}")
        print(f"{indent}    Page {heading['page_number']} | {heading['font_name']} {heading['font_size']:.1f}pt{style_str}")
        print(f"{indent}    Position: ({heading['coordinates']['x1']}, {heading['coordinates']['y1']}) | Confidence: {heading['confidence']:.3f}")
        print()
    
    return outline_data

def print_summary_statistics(all_headings, outline_data):
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    total_headings = len(all_headings)
    pages_with_headings = len(set(h['page_number'] for h in all_headings))
    
    print(f"Total headings found: {total_headings}")
    print(f"Pages with headings: {pages_with_headings}")
    
    level_counts = defaultdict(int)
    for h in all_headings:
        level_counts[h['heading_level']] += 1
    
    print(f"\nHeading level distribution:")
    for level in sorted(level_counts.keys()):
        print(f"   {level}: {level_counts[level]} headings")
    
    font_sizes = [h['font_size'] for h in all_headings if h['font_size'] > 0]
    if font_sizes:
        print(f"\nFont size analysis:")
        print(f"   Average font size: {sum(font_sizes)/len(font_sizes):.1f}pt")
        print(f"   Font size range: {min(font_sizes):.1f}pt - {max(font_sizes):.1f}pt")
    
    bold_count = sum(1 for h in all_headings if h['is_bold'])
    italic_count = sum(1 for h in all_headings if h['is_italic'])
    
    print(f"\nStyle analysis:")
    print(f"   Bold headings: {bold_count} ({bold_count/total_headings*100:.1f}%)")
    print(f"   Italic headings: {italic_count} ({italic_count/total_headings*100:.1f}%)")
    
    page_counts = defaultdict(int)
    for h in all_headings:
        page_counts[h['page_number']] += 1
    
    print(f"\nHeadings per page:")
    for page in sorted(page_counts.keys()):
        print(f"   Page {page}: {page_counts[page]} headings")

class PDFProcessor:
    def __init__(self):
        self.onnx_model_path = "yolo-doclaynet.onnx"
        self.yaml_path = "doclaynet.yaml"
        
        self.detector = None
        
        if os.path.exists(self.onnx_model_path) and os.path.exists(self.yaml_path):
            try:
                self.detector = YOLOv9(self.onnx_model_path, self.yaml_path)
                print("✅ ONNX model loaded successfully for detection")
            except Exception as e:
                print(f"❌ Failed to load ONNX model: {e}")
        else:
            print(f"❌ Model files not found:")
            print(f"   ONNX model: {self.onnx_model_path}")
            print(f"   YAML config: {self.yaml_path}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Files in current directory: {os.listdir('.')}")
    
    def process_pdf(self, pdf_path, output_dir=None):
        if not self.detector:
            print("❌ No model available for processing")
            return None
        
        page_images, page_objects, pdf_document = load_pdf_pages(pdf_path, dpi=150)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        try:
            print("🚀 Starting Hybrid Document Parser...")
            
            all_headings = detect_and_extract_headings(
                self.detector, page_images, page_objects, 
                conf_threshold=0.25, dpi=150, 
                save_visualizations=False, output_dir=output_dir
            )
            
            print(f"\n✅ Successfully found {len(all_headings)} headings across {len(page_images)} pages!")
            
            if all_headings:
                outline_data = generate_document_outline(all_headings)
                print_summary_statistics(all_headings, outline_data)
                
                # Create and save the single JSON structure
                result_json = {
                    "title": base_name,
                    "outline": [
                        {
                            "level": heading["heading_level"],
                            "text": heading["text"],
                            "page": heading["page"]
                        }
                        for heading in outline_data
                    ]
                }
                
                # Save the JSON file
                if output_dir is None:
                    output_dir = Path(__file__).parent / "sample_dataset" / "outputs"
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{base_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, indent=4, ensure_ascii=False)
                
                print(f"💾 Results saved to: {output_file}")
                return output_file
            else:
                print("No headings found to generate outline.")
                return None
        
        finally:
            pdf_document.close()
            print("✅ PDF document closed")
            print("\n🎉 HYBRID DOCUMENT PARSER COMPLETE!")
            print("=" * 60)

def main():
    # Detect if running in Docker vs local environment
    current_dir = Path(__file__).parent
    
    # Check if we're in Docker (look for Docker-specific paths)
    if Path("/app/input").exists() and Path("/app/output").exists():
        # Docker environment
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
        print("🐳 Running in Docker environment")
    else:
        # Local environment
        input_dir = current_dir / "sample_dataset" / "pdfs"
        output_dir = current_dir / "sample_dataset" / "outputs"
        print("💻 Running in local environment")
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = PDFProcessor()
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {input_dir}")
        print(f"Please place PDF files in the '{input_dir}' directory")
        if Path("/app/input").exists():
            print("💡 For Docker: Mount your PDF folder to /app/input")
            print("   Example: docker run -v /path/to/pdfs:/app/input -v /path/to/output:/app/output your-image")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    for pdf_file in pdf_files:
        start_time = time.time()
        
        try:
            output_file = processor.process_pdf(str(pdf_file), output_dir)
            
            processing_time = time.time() - start_time
            print(f"✅ Processed {pdf_file.name} in {processing_time:.2f}s")
            if output_file:
                print(f"📄 Output saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()