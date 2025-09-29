"""
ONNX-based face recognition module.
Uses pre-trained ONNX models for face recognition without InsightFace compilation issues.
"""

import numpy as np
import cv2
import logging
import onnxruntime as ort
from typing import List, Tuple, Optional
import uuid
import urllib.request
import os
from pathlib import Path

class ONNXFaceRecognizer:
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize ONNX face recognizer.
        
        Args:
            similarity_threshold: Threshold for face matching
        """
        self.similarity_threshold = similarity_threshold
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        
        self.load_model()
        
    def load_model(self):
        """Load ONNX face recognition model."""
        try:
            # Try to download a lightweight face recognition model
            model_path = "arcface_r100.onnx"
            
            if not os.path.exists(model_path):
                logging.info("Downloading face recognition model...")
                # Use a publicly available ArcFace model
                model_url = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
                
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    logging.info(f"Downloaded model to {model_path}")
                except Exception as e:
                    logging.warning(f"Could not download model: {e}")
                    # Fall back to simple recognition
                    self._use_simple_recognition()
                    return
            
            # Load ONNX model
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            logging.info(f"Loaded ONNX face recognition model")
            logging.info(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            logging.error(f"Error loading ONNX model: {e}")
            self._use_simple_recognition()
    
    def _use_simple_recognition(self):
        """Fallback to simple recognition method."""
        logging.info("Using simple face recognition fallback")
        self.session = None
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for ONNX model."""
        try:
            # Resize to model input size (typically 112x112 for ArcFace)
            if self.input_shape and len(self.input_shape) >= 3:
                height = self.input_shape[2] if self.input_shape[2] != -1 else 112
                width = self.input_shape[3] if self.input_shape[3] != -1 else 112
            else:
                height, width = 112, 112
            
            # Resize image
            resized = cv2.resize(face_image, (width, height))
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Normalize to [-1, 1] (common for face recognition models)
            normalized = (normalized - 0.5) / 0.5
            
            # Add batch dimension and transpose to NCHW format
            preprocessed = np.transpose(normalized, (2, 0, 1))
            preprocessed = np.expand_dims(preprocessed, axis=0)
            
            return preprocessed
            
        except Exception as e:
            logging.error(f"Error preprocessing face: {e}")
            return None
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using ONNX model.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Face embedding vector or None if failed
        """
        if self.session is None:
            # Use simple fallback method
            return self._generate_simple_embedding(face_image)
        
        try:
            # Preprocess image
            preprocessed = self.preprocess_face(face_image)
            if preprocessed is None:
                return None
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
            embedding = outputs[0][0]  # Remove batch dimension
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error generating ONNX embedding: {e}")
            return self._generate_simple_embedding(face_image)
    
    def _generate_simple_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Simple fallback embedding generation."""
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_image, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram features
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-7)  # Normalize
            
            # Calculate LBP features
            lbp = self._calculate_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
            lbp_hist = lbp_hist.flatten()
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-7)  # Normalize
            
            # Calculate edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Combine features
            embedding = np.concatenate([hist, lbp_hist, [edge_density]])
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error generating simple embedding: {e}")
            return None
    
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        rows, cols = gray_image.shape
        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray_image[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        return lbp
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity."""
        try:
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       known_embeddings: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
        """Find the best matching face from known embeddings."""
        if not known_embeddings:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        try:
            for face_id, known_embedding in known_embeddings:
                similarity = self.compare_embeddings(query_embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = face_id
            
            # Check if best match meets threshold
            if best_similarity >= self.similarity_threshold:
                return best_match_id, best_similarity
            else:
                return None, best_similarity
                
        except Exception as e:
            logging.error(f"Error finding best match: {e}")
            return None, 0.0
    
    def recognize_face(self, face_image: np.ndarray, 
                      known_embeddings: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """Recognize a face by comparing with known embeddings."""
        # Generate embedding for the input face
        embedding = self.generate_embedding(face_image)
        
        if embedding is None:
            return None, 0.0, None
        
        # Find best match
        best_match_id, similarity = self.find_best_match(embedding, known_embeddings)
        
        return best_match_id, similarity, embedding
    
    def generate_face_id(self) -> str:
        """Generate a unique face ID."""
        return f"face_{uuid.uuid4().hex[:8]}"
    
    def is_good_quality_face(self, face_image: np.ndarray, min_quality: float = 0.3) -> bool:
        """Check if face image has sufficient quality for recognition."""
        try:
            if face_image is None or face_image.size == 0:
                return False
            
            # Check size
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                return False
            
            # Check brightness
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 30 or mean_brightness > 220:
                return False
            
            # Check contrast
            contrast = np.std(gray)
            if contrast < 20:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking face quality: {e}")
            return False
    
    def update_similarity_threshold(self, new_threshold: float):
        """Update the similarity threshold for face matching."""
        if 0.0 <= new_threshold <= 1.0:
            self.similarity_threshold = new_threshold
            logging.info(f"Updated similarity threshold to {new_threshold}")
        else:
            logging.warning(f"Invalid threshold value: {new_threshold}")
    
    def get_face_quality_score(self, face_image: np.ndarray) -> float:
        """Assess the quality of a face image for recognition."""
        try:
            if face_image is None or face_image.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Brightness score
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Contrast score
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50.0)
            
            # Sharpness score
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            # Size score
            size_score = min(1.0, min(face_image.shape[:2]) / 64.0)
            
            # Combined score
            quality_score = (brightness_score + contrast_score + sharpness_score + size_score) / 4.0
            
            return quality_score
            
        except Exception as e:
            logging.error(f"Error calculating face quality: {e}")
            return 0.0
