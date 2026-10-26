#!/usr/bin/env python3
"""
Enhanced Live Camera App for Assamese Text Detection and Recognition
With proper Unicode text rendering support
"""

import cv2
import numpy as np
import yaml
from inference_runtime import create_session
from pathlib import Path
from typing import List, Tuple, Optional
import time
import argparse
import logging
from detector import YoloObjectDetector
from PIL import Image, ImageDraw, ImageFont
import os

logger = logging.getLogger(__name__)

class AssameseTextRecognizer:
    """ONNX-based Assamese text recognizer"""
    
    def __init__(self, onnx_path: str, config_path: str, providers=None):
        """Initialize the text recognizer"""
        # Load ONNX model
        self.session = create_session(onnx_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Get model parameters
        self.img_height = self.config['dataset']['valid']['img_height']
        self.img_width = self.config['dataset']['valid']['img_width']
        self.in_channels = self.config['model']['backbone']['in_chans']
        
        # Character set
        self.charset = self.config['charset']['chars']
        self.blank = self.config['charset']['blank']
        self.unknown = self.config['charset']['unknown']
        
        # Build vocabulary
        self.vocab = [self.blank] + list(self.charset) + [self.unknown]
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        print(f"Text Recognizer initialized - Input: [{self.in_channels}, {self.img_height}, {self.img_width}], Vocab: {self.vocab_size}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for text recognition"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image, (self.img_width, self.img_height))
        
        # Normalize to [-1, 1] range (mean=0.5, std=0.5)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor.astype(np.float32)
    
    def ctc_decode(self, logits: np.ndarray) -> str:
        """Decode CTC output to text"""
        # Softmax preserves argmax, so greedy CTC decoding needs only NumPy.
        pred_indices = np.argmax(logits, axis=-1)
        
        # CTC decoding - remove blanks and duplicates
        decoded_indices = []
        prev_idx = -1
        for idx in pred_indices[0]:
            idx_item = idx.item()
            if idx_item != 0 and idx_item != prev_idx:  # Skip blank (0) and duplicates
                decoded_indices.append(idx_item)
            prev_idx = idx_item
        
        # Convert indices to text
        text = ''
        for idx in decoded_indices:
            if idx < len(self.vocab):
                char = self.idx_to_char[idx]
                if char != self.blank:
                    text += char
        
        return text
    
    def recognize(self, image: np.ndarray) -> str:
        """Recognize text in image"""
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]
        
        # Decode text
        text = self.ctc_decode(logits)
        
        return text

class LiveTextDetectionApp:
    """Main application class for live text detection and recognition"""
    
    def __init__(self, detector_model_path: str, recognizer_model_path: str, config_path: str,
                 device: str = 'cuda', translate: bool = False):
        """Initialize the application"""
        print("Initializing Live Text Detection App...")
        
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        # Initialize detector
        self.detector = YoloObjectDetector(
            size=640,
            batched=True,
            path=detector_model_path,
            providers=providers,
            score_threshold=0.25,
            iou_threshold=0.45,
            target_class_id=0  # Assuming class 0 is text/plate
        )
        
        # Initialize text recognizer
        self.recognizer = AssameseTextRecognizer(recognizer_model_path, config_path, providers)
        
        # Initialize translator (Assamese to English)
        self.translator = None
        if translate:
            from offline_translation import OfflineTranslator
            self.translator = OfflineTranslator()
        self.translation_cache = {}  # Cache translations to avoid repeated API calls
        self.last_translation_time = 0
        self.translation_interval = 0.5  # Only translate every 0.5 seconds to avoid slowing down
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_counter = 0
        self.log_interval = 30  # Log every 30 frames to avoid spam
        
        # Set up font path for Assamese text
        self.font_path = None
        self.font_loaded = False
        try:
            # Try to load Lohit Assamese font first
            lohit_font_path = str(Path(__file__).resolve().parent / "font/Lohit-Assamese.ttf")
            if os.path.exists(lohit_font_path):
                # Test if font can actually load
                try:
                    test_font = ImageFont.truetype(lohit_font_path, 24)
                    self.font_path = lohit_font_path
                    self.font_loaded = True
                    print(f"Successfully loaded Lohit Assamese font: {lohit_font_path}")
                except Exception as e:
                    print(f"Failed to load Lohit font: {e}")
            
            # Try fallback fonts if Lohit failed
            if not self.font_loaded:
                fallback_fonts = [
                    "/usr/share/fonts/truetype/lohit-assamese/Lohit-Assamese.ttf",
                    "/usr/share/fonts/truetype/lohit-assamese/lohit_assamese.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
                    "C:/Windows/Fonts/NotoSansBengali.ttf",
                    "C:/Windows/Fonts/NotoSans.ttf",
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/calibri.ttf", 
                    "C:/Windows/Fonts/segoeui.ttf",
                ]
                
                for font_path in fallback_fonts:
                    if os.path.exists(font_path):
                        try:
                            test_font = ImageFont.truetype(font_path, 24)
                            self.font_path = font_path
                            self.font_loaded = True
                            print(f"Successfully loaded fallback font: {os.path.basename(font_path)}")
                            break
                        except Exception as e:
                            print(f"Failed to load {os.path.basename(font_path)}: {e}")
                            continue
                
                if not self.font_loaded:
                    print("Warning: No suitable font found, will use default font")
                
        except Exception as e:
            print(f"Font setup error: {e}")
            self.font_path = None
            self.font_loaded = False
        
        print("App initialized successfully!")
    
    def draw_text(self, frame, text, x, y, lang='as'):
        """
        Draws text on the frame using PIL with a white background.
        """
        try:
            # Convert BGR to RGB for PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            font_size = 24
            
            # Choose font based on language and availability
            if lang == 'en':
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            else:
                # For Assamese text, use the verified font
                if self.font_loaded and self.font_path:
                    try:
                        font = ImageFont.truetype(self.font_path, font_size)
                        font_used = "verified font"
                    except Exception as e:
                        print(f"Failed to load verified font: {e}")
                        font = ImageFont.load_default()
                        font_used = "default"
                else:
                    font = ImageFont.load_default()
                    font_used = "default"
                
                # Debug: Print font being used (only once per session)
                if not hasattr(self, '_font_debug_printed'):
                    print(f"Using font for Assamese text: {font_used}")
                    self._font_debug_printed = True
                
            
            # Get text bounding box
            text_bbox = draw.textbbox((x, y), text, font=font)
            
            # Draw white background rectangle
            background_box = [(text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[3])]
            draw.rectangle(background_box, fill='white')
            
            # Draw text on the white background
            draw.text((x, y), text, font=font, fill='blue')
            
            # Convert back to BGR for OpenCV
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame
            
        except Exception as e:
            print(f"Text drawing error: {e}")
            # Fallback to OpenCV text drawing
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return frame
    
    def translate_text(self, text: str) -> str:
        """Translate text with caching and rate limiting to avoid slowing down"""
        if self.translator is None or not text or not text.strip():
            return ""
        
        # Check cache first
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        # Rate limit translations to avoid slowing down the frame
        current_time = time.time()
        if current_time - self.last_translation_time < self.translation_interval:
            return ""  # Skip translation this frame
        
        started = time.perf_counter()
        try:
            translated = self.translator.translate(text)
            self.translation_cache[text] = translated
            self.last_translation_time = current_time
            logger.info('Translation completed | processing_ms=%.1f | Assamese=%r | English=%r',
                        (time.perf_counter() - started) * 1000, text, translated)
            return translated
        except Exception as e:
            logger.warning('Translation failed | processing_ms=%.1f | error=%s',
                           (time.perf_counter() - started) * 1000, e)
            return ""
    
    def draw_detections(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, texts: List[str],
                        translations: Optional[List[str]] = None) -> np.ndarray:
        """Draw bounding boxes and recognized text on frame"""
        result_frame = frame.copy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(result_frame, f"{float(score):.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw recognized text with Unicode support
            if i < len(texts) and texts[i]:
                # Calculate text position (above the box)
                text_y = max(y1 - 30, 20)
                
                # Use improved text drawing with white background for original text
                result_frame = self.draw_text(result_frame, texts[i], x1, text_y, lang='as')
                
                # Translate and draw English translation below
                if translations is None:
                    translated = self.translate_text(texts[i])
                else:
                    translated = translations[i] if i < len(translations) else None
                if translated:
                    trans_y = text_y - 25  # Position above original text
                    result_frame = self.draw_text(result_frame, translated, x1, trans_y, lang='en')
                
        
        return result_frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 30 / elapsed
            self.fps_start_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process a single frame for detection and recognition"""
        self.frame_counter += 1
        should_log = (self.frame_counter % self.log_interval == 0)
        
        # Detect text regions
        detection_start = time.time()
        boxes, scores = self.detector.detect_one(frame, fit_to_image=True)
        detection_time = time.time() - detection_start
        
        recognized_texts = []
        recognition_times = []
        
        if len(boxes) > 0:
            if should_log:
                logger.debug('Detected %s text regions in %.1f ms', len(boxes), detection_time * 1000)
            
            # Process each detected region
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                # Extract region of interest
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    try:
                        # Recognize text in the region
                        rec_start = time.time()
                        text = self.recognizer.recognize(roi)
                        rec_time = time.time() - rec_start
                        recognition_times.append(rec_time)
                        
                        recognized_texts.append(text)
                        
                        # Log recognition result
                        if should_log:
                            if text.strip():
                                logger.debug('Recognized %r in %.1f ms', text, rec_time * 1000)
                            else:
                                logger.debug('No text recognized (%.1f ms)', rec_time * 1000)
                            
                    except Exception as e:
                        if should_log:
                            logger.warning('Recognition error: %s', e)
                        recognized_texts.append("")
                        recognition_times.append(0)
                else:
                    if should_log:
                        logger.debug('Empty text region')
                    recognized_texts.append("")
                    recognition_times.append(0)
            
            # Log total processing time
            if should_log:
                total_recognition_time = sum(recognition_times)
                logger.debug('Recognition processing_ms=%.1f', total_recognition_time * 1000)
                logger.debug('OCR processing_ms=%.1f', (detection_time + total_recognition_time) * 1000)
        else:
            if should_log:
                logger.debug('No text regions detected')
        
        return boxes, scores, recognized_texts
    
    def run(self, camera_id=0, show_fps: bool = True, gstreamer: bool = False):
        """Run the live camera application"""
        print(f"Starting camera {camera_id}...")
        
        # Initialize camera with timeout
        if gstreamer:
            import re
            if not re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()):
                raise RuntimeError('OpenCV has no GStreamer support. See the CSI camera setup in Readme.md.')
        cap = (cv2.VideoCapture(camera_id, cv2.CAP_GSTREAMER) if gstreamer
               else cv2.VideoCapture(camera_id))
        
        # Set camera properties to 720p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 144)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
        
        # Try to read a frame to test camera
        print("Testing camera connection...")
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"Error: Could not read from camera {camera_id}")
            print("Trying alternative camera settings...")
            
            # Try with different settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, test_frame = cap.read()
            
            if not ret or test_frame is None:
                print(f"Error: Camera {camera_id} is not accessible")
                print("Possible solutions:")
                print("1. Try a different camera ID: --camera 1 or --camera 2")
                print("2. Check if camera is being used by another application")
                print("3. Restart the application")
                cap.release()
                raise RuntimeError(f'Camera {camera_id} is not accessible')
        
        print("Camera started successfully! Resolution: 720p")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                start_time = time.time()
                boxes, scores, texts = self.process_frame(frame)
                process_time = time.time() - start_time
                
                # Draw results
                result_frame = self.draw_detections(frame, boxes, scores, texts)
                
                # Update and display FPS
                self.update_fps()
                if show_fps:
                    cv2.putText(result_frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(result_frame, f"Process: {process_time*1000:.1f}ms", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Assamese Text Detection & Recognition (720p)', result_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"captured_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset FPS counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    print("FPS counter reset")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Live Assamese Text Detection and Recognition (720p)')
    root = Path(__file__).resolve().parent
    parser.add_argument('--detector', type=str, default=str(root / 'det_asm.onnx'),
                       help='Path to detector ONNX model')
    parser.add_argument('--recognizer', type=str, default=str(root / 'model.onnx'),
                       help='Path to text recognizer ONNX model')
    parser.add_argument('--config', type=str, default=str(root / 'config.yaml'),
                       help='Path to config file')
    parser.add_argument('--camera', default='0',
                       help='Camera ID, video path, stream URL, or GStreamer pipeline')
    parser.add_argument('--gstreamer', action='store_true', help='Open --camera as a GStreamer pipeline')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Inference device (default: cuda)')
    parser.add_argument('--translate', action='store_true', help='Enable local ONNX English translation')
    parser.add_argument('--smoke-test', action='store_true', help='Run both models on synthetic inputs without camera or GUI')
    parser.add_argument('--log-interval', type=int, default=30, 
                       help='Sample OCR debug details every N frames; hidden at the default INFO level')
    parser.add_argument('--no-fps', action='store_true', 
                       help='Disable FPS display')
    
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error('--log-interval must be at least 1')
    
    # Check if files exist
    import os
    if not os.path.exists(args.detector):
        print(f"Error: Detector model not found: {args.detector}")
        print("Please provide the correct path to your detector ONNX model")
        return 1
    
    if not os.path.exists(args.recognizer):
        print(f"Error: Recognizer model not found: {args.recognizer}")
        print("Please provide the correct path to your recognizer ONNX model")
        return 1
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    try:
        # Initialize and run the app
        app = LiveTextDetectionApp(args.detector, args.recognizer, args.config,
                                   device=args.device, translate=args.translate and not args.smoke_test)
        app.log_interval = args.log_interval  # Set logging interval
        if args.smoke_test:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            boxes, scores = app.detector.detect_one(frame, fit_to_image=True)
            text = app.recognizer.recognize(np.zeros((32, 128, 3), dtype=np.uint8))
            print(f'Smoke test passed ({args.device}): boxes={boxes.shape}, scores={scores.shape}, text={text!r}')
        else:
            source = int(args.camera) if args.camera.isdecimal() and not args.gstreamer else args.camera
            app.run(camera_id=source, show_fps=not args.no_fps, gstreamer=args.gstreamer)
        return 0
    
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
