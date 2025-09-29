"""
Face detection module using YOLOv8.
Handles face detection and cropping from video frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Tuple, Optional
import os

class FaceDetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLOv8 face detector.
        
        Args:
            model_path: Path to custom YOLO model, if None uses default YOLOv8n
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str = None):
        """Load YOLOv8 model for face detection."""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                logging.info(f"Loaded custom YOLO model from {model_path}")
            else:
                # Use YOLOv8n as default - it will download automatically
                self.model = YOLO('yolov8n.pt')
                logging.info("Loaded default YOLOv8n model")
                
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a frame using YOLOv8.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of tuples (x, y, width, height, confidence) for each detected face
        """
        if self.model is None:
            logging.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter for person class (0) or face class if using face-specific model
                        # For general YOLOv8, we'll detect persons and then crop face region
                        if confidence >= self.confidence_threshold:
                            # Convert to x, y, width, height format
                            x, y = int(x1), int(y1)
                            width, height = int(x2 - x1), int(y2 - y1)
                            
                            # For person detection, estimate face region (upper 1/3 of person)
                            if class_id == 0:  # person class
                                face_height = height // 3
                                face_y = y + height // 10  # slight offset from top
                                face_x = x + width // 4   # center horizontally
                                face_width = width // 2
                                
                                # Ensure face region is within frame bounds
                                face_x = max(0, face_x)
                                face_y = max(0, face_y)
                                face_width = min(face_width, frame.shape[1] - face_x)
                                face_height = min(face_height, frame.shape[0] - face_y)
                                
                                if face_width > 0 and face_height > 0:
                                    faces.append((face_x, face_y, face_width, face_height, float(confidence)))
                            else:
                                # If using a face-specific model, use full detection
                                faces.append((x, y, width, height, float(confidence)))
            
            return faces
            
        except Exception as e:
            logging.error(f"Error during face detection: {e}")
            return []
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face from frame with optional padding.
        
        Args:
            frame: Input image frame
            bbox: Bounding box (x, y, width, height)
            padding: Padding ratio around face
            
        Returns:
            Cropped face image or None if invalid
        """
        try:
            x, y, width, height = bbox
            
            # Add padding
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate padded coordinates
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + width + pad_x)
            y2 = min(frame.shape[0], y + height + pad_y)
            
            # Crop face
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                logging.warning("Empty face crop")
                return None
                
            return face_crop
            
        except Exception as e:
            logging.error(f"Error cropping face: {e}")
            return None
    
    def detect_and_crop_faces(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect faces and return cropped face images with their bounding boxes.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of tuples (cropped_face, bbox, confidence)
        """
        faces = self.detect_faces(frame)
        results = []
        
        for bbox in faces:
            x, y, width, height, confidence = bbox
            face_crop = self.crop_face(frame, (x, y, width, height))
            
            if face_crop is not None:
                results.append((face_crop, (x, y, width, height), confidence))
        
        return results
    
    def draw_detections(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int, float]], 
                       face_ids: List[str] = None) -> np.ndarray:
        """
        Draw face detection boxes on frame.
        
        Args:
            frame: Input image frame
            faces: List of face detections (x, y, width, height, confidence)
            face_ids: Optional list of face IDs to display
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for i, (x, y, width, height, confidence) in enumerate(faces):
            # Draw bounding box
            cv2.rectangle(frame_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Prepare label
            label = f"Face: {confidence:.2f}"
            if face_ids and i < len(face_ids):
                label = f"ID: {face_ids[i]} ({confidence:.2f})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame_copy, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_copy
    
    def preprocess_for_recognition(self, face_image: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Preprocess face image for recognition model.
        
        Args:
            face_image: Cropped face image
            target_size: Target size for recognition model
            
        Returns:
            Preprocessed face image
        """
        try:
            # Resize to target size
            resized = cv2.resize(face_image, target_size)
            
            # Convert BGR to RGB if needed
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logging.error(f"Error preprocessing face for recognition: {e}")
            return None
    
    def is_valid_face(self, face_image: np.ndarray, min_size: int = 50) -> bool:
        """
        Check if detected face is valid for processing.
        
        Args:
            face_image: Cropped face image
            min_size: Minimum face size in pixels
            
        Returns:
            True if face is valid, False otherwise
        """
        if face_image is None or face_image.size == 0:
            return False
        
        height, width = face_image.shape[:2]
        
        # Check minimum size
        if height < min_size or width < min_size:
            return False
        
        # Check aspect ratio (faces should be roughly square to rectangular)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        # Check if image is too dark or too bright
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        mean_intensity = np.mean(gray)
        if mean_intensity < 20 or mean_intensity > 235:
            return False
        
        return True
