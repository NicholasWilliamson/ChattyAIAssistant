
#!/usr/bin/env python3
"""
Enhanced Human Detection System with Debug Mode
Detects people using multiple methods with debugging information
"""

import cv2
import numpy as np
import face_recognition
import pickle
import json
import datetime
import logging
import requests
from picamera2 import Picamera2
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityHumanDetector:
    def __init__(self):
        self.debug_mode = True  # Enable debug output
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_face_encodings()
        
        # Initialize camera with proper display configuration
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        logger.info("Camera configuration has been adjusted!")
        
        # Initialize detection methods
        self.init_detection_methods()
        
        # Load Telegram config
        self.load_telegram_config()
        
    def load_face_encodings(self):
        """Load known face encodings"""
        try:
            with open('face_encodings.pkl', 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
        except FileNotFoundError:
            logger.warning("No face encodings file found. Face recognition disabled.")
    
    def init_detection_methods(self):
        """Initialize all detection methods"""
        # Try to load YOLO
        self.yolo_net = None
        self.yolo_output_layers = None
        
        try:
            self.yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            layer_names = self.yolo_net.getLayerNames()
            self.yolo_output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            logger.info("YOLO loaded successfully")
        except Exception as e:
            logger.warning(f"YOLO not available: {e}. Using backup detection methods.")
        
        # Initialize HOG detector (more sensitive)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize Haar cascades
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            logger.info("Haar cascades loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Haar cascades: {e}")
    
    def load_telegram_config(self):
        """Load Telegram configuration"""
        try:
            with open('telegram_config.json', 'r') as f:
                config = json.load(f)
                self.telegram_token = config.get('bot_token')
                self.telegram_chat_id = config.get('chat_id')
            logger.info("Telegram configuration loaded")
        except FileNotFoundError:
            logger.warning("Telegram config not found. Notifications disabled.")
            self.telegram_token = None
            self.telegram_chat_id = None
    
    def detect_with_yolo(self, frame):
        """Detect people using YOLO"""
        if self.yolo_net is None:
            return []
        
        try:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.yolo_net.setInput(blob)
            outputs = self.yolo_net.forward(self.yolo_output_layers)
            
            boxes = []
            confidences = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if class_id == 0 and confidence > 0.3:  # Lower threshold for better detection
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            detections = []
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    detections.append((x, y, w, h, confidences[i]))
                    if self.debug_mode:
                        logger.info(f"YOLO detected person: confidence={confidences[i]:.2f}")
            
            return detections
        except Exception as e:
            if self.debug_mode:
                logger.error(f"YOLO detection error: {e}")
            return []
    
    def detect_with_hog(self, frame):
        """Detect people using HOG descriptor - more sensitive"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect people with multiple scales and lower threshold
            detections = []
            
            # Try multiple detection parameters for better sensitivity
            for scale in [1.03, 1.05, 1.1]:  # Different scales
                for padding in [(4, 4), (8, 8), (16, 16)]:  # Different padding
                    try:
                        rects, weights = self.hog.detectMultiScale(
                            gray, 
                            winStride=(4, 4), 
                            padding=padding, 
                            scale=scale,
                            hitThreshold=0.0,  # Lower threshold for better detection
                            groupThreshold=1
                        )
                        
                        for i, (x, y, w, h) in enumerate(rects):
                            confidence = weights[i] if i < len(weights) else 0.5
                            detections.append((x, y, w, h, confidence))
                            if self.debug_mode:
                                logger.info(f"HOG detected person: confidence={confidence:.2f}, scale={scale}")
                    except Exception as e:
                        if self.debug_mode:
                            logger.debug(f"HOG detection failed with scale {scale}: {e}")
                        continue
            
            # Remove duplicates
            if detections:
                detections = self.remove_duplicate_detections(detections)
                
            return detections
        except Exception as e:
            if self.debug_mode:
                logger.error(f"HOG detection error: {e}")
            return []
    
    def detect_with_cascades(self, frame):
        """Detect using Haar cascades"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Face detection - lower scale factor for better detection
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # Lower for better detection
                minNeighbors=3,    # Lower for more sensitivity
                minSize=(20, 20)   # Smaller minimum size
            )
            
            for (x, y, w, h) in faces:
                detections.append((x, y, w, h, 0.8, "face"))
                if self.debug_mode:
                    logger.info(f"Face detected at ({x}, {y}, {w}, {h})")
            
            # Upper body detection
            bodies = self.upper_body_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in bodies:
                detections.append((x, y, w, h, 0.7, "upper_body"))
                if self.debug_mode:
                    logger.info(f"Upper body detected at ({x}, {y}, {w}, {h})")
            
            # Full body detection
            full_bodies = self.body_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3,
                minSize=(50, 50)
            )
            
            for (x, y, w, h) in full_bodies:
                detections.append((x, y, w, h, 0.6, "full_body"))
                if self.debug_mode:
                    logger.info(f"Full body detected at ({x}, {y}, {w}, {h})")
                    
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Cascade detection error: {e}")
        
        return detections
    
    def detect_with_face_recognition(self, frame):
        """Detect faces using face_recognition library"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations with different models
            detections = []
            
            # Try HOG model (faster)
            try:
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                for (top, right, bottom, left) in face_locations:
                    w = right - left
                    h = bottom - top
                    detections.append((left, top, w, h, 0.9, "face_recognition_hog"))
                    if self.debug_mode:
                        logger.info(f"Face recognition (HOG) detected face at ({left}, {top}, {w}, {h})")
            except Exception as e:
                if self.debug_mode:
                    logger.debug(f"Face recognition HOG failed: {e}")
            
            # Try CNN model (more accurate but slower)
            try:
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
                for (top, right, bottom, left) in face_locations:
                    w = right - left
                    h = bottom - top
                    detections.append((left, top, w, h, 0.95, "face_recognition_cnn"))
                    if self.debug_mode:
                        logger.info(f"Face recognition (CNN) detected face at ({left}, {top}, {w}, {h})")
            except Exception as e:
                if self.debug_mode:
                    logger.debug(f"Face recognition CNN failed: {e}")
            
            return detections
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Face recognition error: {e}")
            return []
    
    def remove_duplicate_detections(self, detections, overlap_threshold=0.3):
        """Remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4] if len(x) > 4 else 0.5, reverse=True)
        
        filtered = []
        for detection in detections:
            x1, y1, w1, h1 = detection[:4]
            
            # Check overlap with already accepted detections
            overlap = False
            for accepted in filtered:
                x2, y2, w2, h2 = accepted[:4]
                
                # Calculate intersection over union
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                
                if xi2 > xi1 and yi2 > yi1:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    union = w1 * h1 + w2 * h2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > overlap_threshold:
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(detection)
        
        return filtered
    
    def send_telegram_alert(self, message, image_path=None):
        """Send Telegram notification"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if image_path:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {'chat_id': self.telegram_chat_id}
                    requests.post(url, data=data, files=files, timeout=10)
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")
            return False
    
    def run_detection(self):
        """Main detection loop with debug information"""
        print("🛡️ Security Human Detection Application (DEBUG MODE)")
        print("==================================================")
        print("This system detects people using multiple methods!")
        print("DEBUG MODE: Detailed detection information will be shown")
        print("Press 'q' to quit, 's' to save current frame, 'd' to toggle debug")
        
        self.picam2.start()
        logger.info("Camera started")
        logger.info("Starting Enhanced Security Person Detection System...")
        
        # Wait for camera to warm up
        time.sleep(2)
        
        frame_count = 0
        detection_count = 0
        last_detection_time = 0
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                # Note: Camera is already configured for RGB888, so no conversion needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_count += 1
                
                if self.debug_mode and frame_count % 30 == 0:  # Debug info every 30 frames
                    logger.info(f"Processed {frame_count} frames, {detection_count} detections")
                
                # Combine all detection methods
                all_detections = []
                
                # YOLO detection
                yolo_detections = self.detect_with_yolo(frame)
                all_detections.extend([(x, y, w, h, conf, "YOLO") for x, y, w, h, conf in yolo_detections])
                
                # HOG detection
                hog_detections = self.detect_with_hog(frame)
                all_detections.extend([(x, y, w, h, conf, "HOG") for x, y, w, h, conf in hog_detections])
                
                # Cascade detection
                cascade_detections = self.detect_with_cascades(frame)
                all_detections.extend(cascade_detections)
                
                # Face recognition detection
                face_detections = self.detect_with_face_recognition(frame)
                all_detections.extend(face_detections)
                
                # Remove duplicates
                unique_detections = self.remove_duplicate_detections(all_detections)
                
                # Draw detections
                current_time = time.time()
                for detection in unique_detections:
                    x, y, w, h = detection[:4]
                    confidence = detection[4] if len(detection) > 4 else 0.5
                    method = detection[5] if len(detection) > 5 else "Unknown"
                    
                    detection_count += 1
                    
                    # Draw bounding box
                    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"{method}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Send alert (limit to once per 5 seconds)
                    if current_time - last_detection_time > 5:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        message = f"🚨 Person detected at {timestamp} using {method} (confidence: {confidence:.2f})"
                        
                        # Save detection image
                        detection_filename = f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(detection_filename, frame)
                        
                        # Send Telegram alert
                        self.send_telegram_alert(message, detection_filename)
                        
                        logger.info(f"DETECTION: {message}")
                        last_detection_time = current_time
                
                # Add status overlay
                status_text = f"Detections: {len(unique_detections)} | Frame: {frame_count} | Debug: {'ON' if self.debug_mode else 'OFF'}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame with error handling
                try:
                    cv2.imshow('🎥 Enhanced Security Camera - Multiple Detection Methods', frame)
                    cv2.waitKey(1)  # Process window events
                except Exception as e:
                    if self.debug_mode:
                        logger.warning(f"Display error (detection still working): {e}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"manual_save_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved as {filename}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
                
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Detection error: {e}")
        finally:
            self.picam2.stop()
            logger.info("Camera stopped")
            cv2.destroyAllWindows()
            logger.info(f"Security monitoring completed. Total detections: {detection_count}")
            
            # Clean up camera resources
            try:
                self.picam2.close()
                logger.info("Camera closed successfully.")
            except Exception as e:
                logger.warning(f"Camera cleanup warning: {e}")

def main():
    detector = SecurityHumanDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()